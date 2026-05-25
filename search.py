"""
Drop-in replacement for the RAG helper module in the uploaded code.

What changed, while keeping the same public entry points:
- cleaner snippets/context from noisy HTML-ish text
- dense oversampling + query-aware reranking
- optional local CrossEncoder reranking if config.RERANKER_MODEL is set
- multi-query retrieval in chat_rag with reciprocal-rank fusion
- fixed router/history message roles
- fixed threshold handling, doc merging, citation source emission, and timeline returns

Optional config knobs you can add without breaking existing config.py:
    ENABLE_MULTI_QUERY = True
    MAX_SEARCH_QUERIES = 4
    SEARCH_FETCH_MULTIPLIER = 8
    CONTEXT_MAX_CHARS = 14000
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # or None
    EMBEDDING_QUERY_PREFIX = None  # set to "query: " if your index was built for E5-style prompts
    SEARCH_VERSION = "rag-v2"
"""

from __future__ import annotations

import faiss
import hashlib
import html
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
from typing import Any, Dict, Generator, Iterable, Iterator, List, Sequence, Set

import numpy as np
import pandas as pd
import streamlit as st

import config
from config import INDEX_PATH, MAP_PATH, PAGES_PATH
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



SPINNER_MESSAGES = [
    "Tuning quantum flux capacitors...",
    "Feeding PDFs to our pet AI llama...",
    "Consulting the Book of Infinite Wisdom...",
    "Summoning digital gnomes for indexing...",
    "Injecting caffeine into vector space...",
    "Decoding ancient neural runes...",
    "Polishing the FAISS crystal ball...",
    "Massaging cosine similarities...",
    "Bribing the embeddings to behave...",
    "Unleashing the power of dot products...",
]


# ---------------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------------

def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


SEARCH_VERSION = str(_cfg("SEARCH_VERSION", "rag-v2.1"))
DEFAULT_FETCH_MULTIPLIER = int(_cfg("SEARCH_FETCH_MULTIPLIER", 8))
DEFAULT_CONTEXT_MAX_CHARS = int(_cfg("CONTEXT_MAX_CHARS", 14_000))
DEFAULT_MAX_SEARCH_QUERIES = int(_cfg("MAX_SEARCH_QUERIES", 4))
ENABLE_MULTI_QUERY = bool(_cfg("ENABLE_MULTI_QUERY", True))
RRF_K = int(_cfg("RRF_K", 60))
DEFAULT_HISTORY_TURNS = int(_cfg("RAG_HISTORY_TURNS", 8))

# Loaded eagerly inside initialize_search_index; accessed by _rerank_results and search_pdfs.
_reranker = None
_bm25_cache: tuple | None = None  # (BM25Okapi, list[doc_id])


# ---------------------------------------------------------------------------
# Text cleaning and snippet construction
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(accept all|reject all|cookie settings|manage cookies)\s*$", re.I),
    re.compile(r"\b(cookie policy|privacy policy|terms of use|accessibility statement)\b", re.I),
    re.compile(r"\b(subscribe to our newsletter|follow us on|share this|all rights reserved)\b", re.I),
    re.compile(r"\b(skip to main content|download pdf|print this page)\b", re.I),
    re.compile(r"^\s*(home|about|contact|press|publications|events|jobs)\s*$", re.I),
    re.compile(r"^\s*(facebook|twitter|x|linkedin|instagram|youtube)\s*$", re.I),
]

_TAG_RX = re.compile(r"(?is)<\s*/?\s*[a-z][^>]*>")
_SCRIPT_STYLE_RX = re.compile(r"(?is)<(script|style|noscript|svg|header|footer|nav)[^>]*>.*?</\1>")
_WS_RX = re.compile(r"\s+")
_SENTENCE_RX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TOKEN_RX = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.-]{1,}")

_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "among", "because", "been",
    "before", "being", "between", "both", "could", "does", "doing", "down", "during",
    "each", "from", "further", "have", "having", "here", "into", "itself", "just",
    "more", "most", "other", "over", "same", "should", "some", "such", "than", "that",
    "their", "there", "these", "they", "this", "those", "through", "under", "until",
    "very", "what", "when", "where", "which", "while", "with", "would", "your", "transport",
    "environment",
}

_ACRONYM_EXPANSIONS = {
    "uco": "used cooking oil",
    "iluc": "indirect land use change",
    "ets": "emissions trading system",
    "cbam": "carbon border adjustment mechanism",
    "co2": "carbon dioxide",
    "ghg": "greenhouse gas",
    "hev": "hybrid electric vehicle",
    "bev": "battery electric vehicle",
    "phev": "plug-in hybrid electric vehicle",
    "saf": "sustainable aviation fuel",
    "rfnbos": "renewable fuels of non-biological origin",
}


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = _SCRIPT_STYLE_RX.sub(" ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = _TAG_RX.sub(" ", text)
    return text


def clean_text(text: str, *, max_chars: int | None = None) -> str:
    """Clean noisy extracted page/chunk text without changing meaning."""
    if not text:
        return ""

    text = _strip_html(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"-\s*\n\s*", "", text)  # de-hyphenate line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    lines: list[str] = []
    seen_recent: list[str] = []
    for raw_line in text.splitlines():
        line = _WS_RX.sub(" ", raw_line).strip(" \t\r\n-|•")
        if not line:
            continue
        if any(rx.search(line) for rx in _BOILERPLATE_PATTERNS):
            continue
        # Drop tiny nav fragments, but keep short technical acronyms like UCO/ETS/CO2.
        if len(line) < 3 and not line.isupper():
            continue
        norm = line.lower()
        if norm in seen_recent:
            continue
        seen_recent.append(norm)
        if len(seen_recent) > 12:
            seen_recent.pop(0)
        lines.append(line)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = _WS_RX.sub(" ", cleaned).strip()

    if max_chars and len(cleaned) > max_chars:
        return cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return cleaned


def _tokenize(text: str) -> list[str]:
    tokens = [t.lower().strip("._-") for t in _TOKEN_RX.findall(text or "")]
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _query_terms(query: str) -> set[str]:
    terms = set(_tokenize(query))
    for t in list(terms):
        expansion = _ACRONYM_EXPANSIONS.get(t)
        if expansion:
            terms.update(_tokenize(expansion))
    return terms


def _split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = _SENTENCE_RX.split(text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > 900:
            # Bad PDF extraction can produce page-long pseudo-sentences.
            sentences.extend([p.strip() for p in re.split(r"\s{2,}|;\s+", part) if p.strip()])
        else:
            sentences.append(part)
    return sentences


def _sentence_score(sentence: str, terms: set[str]) -> float:
    if not sentence or not terms:
        return 0.0
    toks = _tokenize(sentence)
    if not toks:
        return 0.0
    counts = defaultdict(int)
    for tok in toks:
        counts[tok] += 1
    matched = sum(min(counts[t], 3) for t in terms if t in counts)
    coverage = sum(1 for t in terms if t in counts) / max(1, len(terms))
    density = matched / math.sqrt(len(toks))
    return density + coverage


def make_query_focused_snippet(
    raw_text: str,
    query: str,
    *,
    max_chars: int = 700,
    fallback_start: int = 0,
) -> str:
    """Return compact, query-centered evidence rather than a blind substring."""
    cleaned = clean_text(raw_text, max_chars=max(max_chars * 4, 2_000))
    if not cleaned:
        return ""

    terms = _query_terms(query)
    sentences = _split_sentences(cleaned)
    if not sentences:
        return cleaned[:max_chars].rsplit(" ", 1)[0].strip()

    scored = [(i, _sentence_score(s, terms), s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: (x[1], -abs(x[0] - fallback_start)), reverse=True)

    selected_idx: set[int] = set()
    for i, score, _ in scored[:8]:
        if score <= 0 and selected_idx:
            continue
        selected_idx.add(i)
        # Add one neighboring sentence for context when budget allows.
        if i > 0:
            selected_idx.add(i - 1)
        if i + 1 < len(sentences):
            selected_idx.add(i + 1)
        candidate = " ".join(sentences[j] for j in sorted(selected_idx))
        if len(candidate) >= max_chars * 0.75:
            break

    if not selected_idx:
        selected_idx.add(0)

    snippet = " ".join(sentences[i] for i in sorted(selected_idx))
    snippet = clean_text(snippet)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0].strip()
    return snippet


# ---------------------------------------------------------------------------
# Loading artefacts
# ---------------------------------------------------------------------------

def _parse_datetime(date_value: Any) -> datetime | None:
    if date_value is None or (isinstance(date_value, float) and np.isnan(date_value)):
        return None
    if isinstance(date_value, datetime):
        return date_value.replace(tzinfo=None)

    s = str(date_value).strip()
    if not s:
        return None

    # Fast path for common values in the current codebase.
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%B %d, %Y, %I:%M:%S %p",
        "%B %d, %Y",
        "%b %d, %Y, %I:%M:%S %p",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue

    try:
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().replace(tzinfo=None)
    except Exception:
        return None


def _parse_year(date_value: Any) -> int | None:
    dt = _parse_datetime(date_value)
    return dt.year if dt else None


def _make_embeddings(openai_api_key: str | None = None):
    model_name = config.EMBEDDING_MODEL
    if "text-embedding" in model_name:
        return OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
    # Normalizing local embeddings is usually the right choice for cosine-style FAISS indexes.
    # If the existing index was built without normalization, set EMBEDDING_NORMALIZE=False.
    encode_kwargs = {}
    if hasattr(config, "EMBEDDING_NORMALIZE"):
        encode_kwargs["normalize_embeddings"] = bool(config.EMBEDDING_NORMALIZE)
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)


@st.cache_resource(show_spinner=random.choice(SPINNER_MESSAGES))
def initialize_search_index(openai_api_key: str | None = None):
    """
    Returns
    -------
    index     : faiss.Index
    embeddings: Embedding model (HF or OpenAI)
    mapping   : pd.DataFrame  (index on vector_id)
    pages     : pd.DataFrame  (index on document ID)
    year2vec  : dict[int, np.ndarray]

    Side effects: sets module-level _reranker and _bm25_cache singletons.
    """
    global _reranker, _bm25_cache

    print("Loading embedding model...")
    embeddings = _make_embeddings(openai_api_key=openai_api_key)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading Parquet files...")
    mapping = pd.read_parquet(MAP_PATH)
    pages = pd.read_parquet(PAGES_PATH)

    if "vector_id" not in mapping.columns:
        raise ValueError("MAP_PATH parquet must contain a 'vector_id' column")
    if "ID" not in pages.columns:
        raise ValueError("PAGES_PATH parquet must contain an 'ID' column")

    mapping = mapping.drop_duplicates(subset=["vector_id"]).set_index("vector_id", drop=True)
    pages = pages.drop_duplicates(subset=["ID"]).set_index("ID", drop=True)

    if "Publication Date" in pages.columns:
        pages["year"] = pages["Publication Date"].apply(_parse_year)
    else:
        pages["year"] = None

    tmp = (
        mapping.reset_index()
        .merge(pages[["year"]], left_on="doc_id", right_index=True, how="left")
        .dropna(subset=["year"])
    )
    year2vec = {
        int(y): np.ascontiguousarray(grp["vector_id"].to_numpy(np.int64))
        for y, grp in tmp.groupby("year", sort=True)
    }

    print("Loading reranker model...")
    reranker_model = _cfg("RERANKER_MODEL", None)
    if reranker_model:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(reranker_model)
        except Exception as e:
            print(f"Reranker load failed ({e}); reranking disabled.")

    bm25_path = _cfg("BM25_PATH", "./embeddings/bm25_index.pkl")
    try:
        import pickle, os
        from rank_bm25 import BM25Okapi
        if os.path.exists(bm25_path):
            print("Loading pre-built BM25 index...")
            with open(bm25_path, "rb") as f:
                _bm25_cache = pickle.load(f)
        else:
            print("Building BM25 index (will save to disk for next run)...")
            doc_ids = list(pages.index)
            corpus = [_tokenize(str(pages.loc[did].get("fulltext") or "")) for did in doc_ids]
            _bm25_cache = (BM25Okapi(corpus), doc_ids)
            with open(bm25_path, "wb") as f:
                pickle.dump(_bm25_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"BM25 load/build failed ({e}); hybrid search disabled.")

    print("Successfully loaded search artefacts")
    return index, embeddings, mapping, pages, year2vec


# ---------------------------------------------------------------------------
# Search scoring and merging
# ---------------------------------------------------------------------------

def _as_float32_2d(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.ascontiguousarray(arr)


def _embedding_query_text(query: str) -> str:
    query = clean_text(query)
    prefix = _cfg("EMBEDDING_QUERY_PREFIX", None)
    if prefix is None:
        model_name = str(_cfg("EMBEDDING_MODEL", "")).lower()
        # E5-family models expect "query: ..." for queries. OpenAI embeddings do not.
        prefix = "query: " if "e5" in model_name else ""
    return f"{prefix}{query}".strip()


def _metric_similarity(faiss_index, raw_score: float) -> float:
    metric_type = getattr(faiss_index, "metric_type", None)
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        # If vectors are normalized, this is cosine similarity. Keep negative values possible;
        # threshold defaults to 0, so poor opposite-direction matches drop out.
        return float(raw_score)
    if metric_type == faiss.METRIC_L2:
        return float(1.0 / (1.0 + max(raw_score, 0.0)))
    return float(1.0 / (1.0 + abs(raw_score)))


def _faiss_search(
    faiss_index,
    q_vec: Sequence[float],
    k: int,
    *,
    allowed_vec_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    q = _as_float32_2d(q_vec)
    k = max(1, int(k))

    if allowed_vec_ids is not None and len(allowed_vec_ids) == 0:
        return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

    if allowed_vec_ids is not None:
        allowed_vec_ids = np.ascontiguousarray(allowed_vec_ids.astype(np.int64))
        try:
            try:
                selector = faiss.IDSelectorArray(allowed_vec_ids)
            except TypeError:
                selector = faiss.IDSelectorArray(len(allowed_vec_ids), faiss.swig_ptr(allowed_vec_ids))
            params = faiss.SearchParametersIVF(sel=selector)
            return faiss_index.search(q, k, params=params)
        except Exception:
            # Not all FAISS index types accept SearchParametersIVF. Fall back to oversampling
            # and post-filtering. This is slower, but it preserves functionality.
            pass

    if allowed_vec_ids is None:
        return faiss_index.search(q, k)

    oversample = min(max(k * 50, k + 200), int(getattr(faiss_index, "ntotal", k * 50)))
    D, I = faiss_index.search(q, oversample)
    allowed = set(int(x) for x in allowed_vec_ids)
    pairs = [(float(d), int(i)) for d, i in zip(D[0], I[0]) if int(i) in allowed and int(i) >= 0]
    pairs = pairs[:k]
    if not pairs:
        return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
    d_out = np.asarray([[p[0] for p in pairs]], dtype=np.float32)
    i_out = np.asarray([[p[1] for p in pairs]], dtype=np.int64)
    return d_out, i_out


def _doc_key(hit: dict) -> str:
    for field in ("pdf_url", "url", "filename", "title", "doc_id"):
        value = hit.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return hashlib.sha256(str(hit).encode("utf-8", errors="ignore")).hexdigest()


def _lexical_score(query: str, text: str, title: str | None = None) -> float:
    terms = _query_terms(query)
    if not terms:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    token_set = set(text_tokens)
    title_set = set(_tokenize(title or ""))
    coverage = sum(1 for t in terms if t in token_set) / max(1, len(terms))
    title_coverage = sum(1 for t in terms if t in title_set) / max(1, len(terms))
    phrase_bonus = 0.25 if clean_text(query).lower() in clean_text(text).lower() else 0.0
    return min(1.0, coverage + 0.4 * title_coverage + phrase_bonus)


def _date_weight(publication_date: Any, alpha: float) -> tuple[float, str | None]:
    dt = _parse_datetime(publication_date)
    if not dt:
        return 1.0, str(publication_date).strip() if publication_date else None
    days_diff = max(0, (datetime.now() - dt).days)
    weight = float(np.exp(-float(alpha) * days_diff / 365.0)) if alpha else 1.0
    return weight, dt.strftime("%Y-%m-%d")


def _safe_loc(df: pd.DataFrame, key: Any):
    try:
        row = df.loc[key]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def _make_result_from_hit(
    *,
    query: str,
    faiss_index,
    dist: float,
    vec_id: int,
    mapping_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    alpha: float,
    max_snippet_length: int,
    year_filtered: bool,
) -> dict | None:
    if vec_id < 0:
        return None

    meta_row = _safe_loc(mapping_df, vec_id)
    if meta_row is None:
        return None

    doc_id = meta_row.get("doc_id")
    doc_row = _safe_loc(pages_df, doc_id)
    if doc_row is None:
        return None

    full_text = str(doc_row.get("fulltext") or "")
    start = int(meta_row.get("start_char", 0) or 0)
    end = int(meta_row.get("end_char", start) or start)
    if end < start:
        start, end = end, start

    window_chars = max(max_snippet_length * 3, 1_800)
    window_start = max(0, start - window_chars // 3)
    window_end = min(len(full_text), max(end + window_chars // 3, start + window_chars))
    raw_window = full_text[window_start:window_end]

    snippet = make_query_focused_snippet(
        raw_window,
        query,
        max_chars=max_snippet_length,
        fallback_start=max(0, start - window_start),
    )
    if not snippet:
        return None

    similarity = _metric_similarity(faiss_index, float(dist))
    date_weight, pub_date_clean = _date_weight(doc_row.get("Publication Date"), alpha)
    if year_filtered:
        date_weight = 1.0

    title = doc_row.get("Title")
    lexical = _lexical_score(query, snippet, str(title or ""))

    # Dense score remains the anchor. Lexical score helps acronyms, regulation names,
    # exact dates, and messy text where embeddings can be too smooth.
    weighted = (similarity * date_weight) + (0.12 * lexical)

    return {
        "title": title,
        "filename": doc_row.get("PDF Filename"),
        "summary": doc_row.get("Summary"),
        "publication_date": pub_date_clean,
        "publication_type": doc_row.get("Publication Type"),
        "url": doc_row.get("Article URL"),
        "pdf_url": doc_row.get("PDF URL"),
        "snippet": snippet,
        "score": float(similarity),
        "date_weight": float(date_weight),
        "lexical_score": float(lexical),
        "weighted_score": float(weighted),
        "combined_score": float(weighted),
        "start_char": start,
        "end_char": end,
        "doc_id": doc_id,
        "vector_id": int(vec_id),
    }


def _overlaps(a: dict, b: dict, *, min_gap: int = 180) -> bool:
    if a.get("doc_id") != b.get("doc_id"):
        return False
    a0, a1 = int(a.get("start_char", 0)), int(a.get("end_char", 0))
    b0, b1 = int(b.get("start_char", 0)), int(b.get("end_char", 0))
    return not (a1 + min_gap < b0 or b1 + min_gap < a0)


def merge_snippets(
    hits: list[dict],
    *,
    max_snippets_per_doc: int = 3,
    joiner: str = " ... ",
) -> list[dict]:
    """Merge chunk hits into one result per document, preserving best score and readable order."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        buckets[_doc_key(h)].append(h)

    merged: list[dict] = []
    for _, bucket in buckets.items():
        # Pick evidence by score, avoiding near-duplicate neighboring chunks.
        bucket.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
        selected: list[dict] = []
        seen_snippets: set[str] = set()
        for item in bucket:
            norm_snip = clean_text(item.get("snippet", "")).lower()[:240]
            if not norm_snip or norm_snip in seen_snippets:
                continue
            if any(_overlaps(item, chosen) for chosen in selected):
                continue
            selected.append(item)
            seen_snippets.add(norm_snip)
            if len(selected) >= max_snippets_per_doc:
                break

        if not selected:
            continue

        selected_for_text = sorted(selected, key=lambda x: int(x.get("start_char", 0)))
        snippets_combined = joiner.join(s["snippet"] for s in selected_for_text if s.get("snippet"))
        best = max(bucket, key=lambda x: x.get("weighted_score", 0.0)).copy()
        best["snippet"] = snippets_combined
        best["combined_score"] = float(
            max(s.get("weighted_score", 0.0) for s in selected)
            + 0.03 * min(len(selected), max_snippets_per_doc)
        )
        best["evidence_count"] = len(selected)
        merged.append(best)

    merged.sort(key=itemgetter("combined_score"), reverse=True)
    return merged


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _rerank_results(query: str, docs: list[dict], *, limit: int) -> list[dict]:
    if not docs:
        return []
    if _reranker is None:
        return docs[:limit]

    pairs = [(query, f"{d.get('title') or ''}\n{d.get('snippet') or ''}") for d in docs]
    try:
        raw_scores = _reranker.predict(pairs)
        raw_scores = [float(x) for x in np.asarray(raw_scores).reshape(-1).tolist()]
    except Exception:
        return docs[:limit]

    norm_scores = _minmax(raw_scores)
    reranked: list[dict] = []
    for doc, rr_raw, rr_norm in zip(docs, raw_scores, norm_scores):
        item = doc.copy()
        item["reranker_score"] = float(rr_raw)
        item["reranker_score_norm"] = float(rr_norm)
        item["combined_score"] = (
            0.65 * float(item.get("combined_score", item.get("weighted_score", 0.0)))
            + 0.30 * rr_norm
            + 0.05 * float(item.get("lexical_score", 0.0))
        )
        reranked.append(item)

    reranked.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return reranked[:limit]


def _bm25_search(
    query: str,
    pages_df: pd.DataFrame,
    *,
    k: int,
    alpha: float,
    max_snippet_length: int,
) -> list[dict]:
    if _bm25_cache is None:
        return []
    bm25, doc_ids = _bm25_cache
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    fetch = min(k * 2, len(doc_ids))
    top_indices = np.argpartition(scores, -fetch)[-fetch:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    max_score = float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0
    if max_score <= 0:
        return []

    results: list[dict] = []
    for idx in top_indices:
        raw_score = float(scores[idx])
        if raw_score <= 0:
            break
        norm_score = raw_score / max_score

        doc_id = doc_ids[idx]
        doc_row = _safe_loc(pages_df, doc_id)
        if doc_row is None:
            continue

        full_text = str(doc_row.get("fulltext") or "")
        snippet = make_query_focused_snippet(full_text, query, max_chars=max_snippet_length)
        if not snippet:
            continue

        date_weight, pub_date_clean = _date_weight(doc_row.get("Publication Date"), alpha)
        title = doc_row.get("Title")
        lexical = _lexical_score(query, snippet, str(title or ""))
        weighted = (norm_score * date_weight) + (0.12 * lexical)

        results.append({
            "title": title,
            "filename": doc_row.get("PDF Filename"),
            "summary": doc_row.get("Summary"),
            "publication_date": pub_date_clean,
            "publication_type": doc_row.get("Publication Type"),
            "url": doc_row.get("Article URL"),
            "pdf_url": doc_row.get("PDF URL"),
            "snippet": snippet,
            "score": norm_score,
            "date_weight": float(date_weight),
            "lexical_score": float(lexical),
            "weighted_score": float(weighted),
            "combined_score": float(weighted),
            "doc_id": doc_id,
            "vector_id": -1,
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:k]


def search_pdfs(
    query: str,
    faiss_index,
    embeddings,
    mapping_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    *,
    k: int = config.SEARCH_RESULT_K,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    threshold: float = 0.0,
    year2vec: dict[int, np.ndarray] | None = None,
    year: int | None = None,
) -> list[dict]:
    """Hybrid (FAISS + BM25) search with RRF fusion and optional CrossEncoder reranking."""
    query = clean_text(query)
    if not query:
        return []

    fetch_multiplier = max(1, int(_cfg("SEARCH_FETCH_MULTIPLIER", DEFAULT_FETCH_MULTIPLIER)))
    fetch_k = max(int(k) * fetch_multiplier, int(k) + 20)

    q_vec = embeddings.embed_query(_embedding_query_text(query))

    allowed_vec_ids = None
    if year is not None and year2vec:
        allowed_vec_ids = year2vec.get(int(year))
        if allowed_vec_ids is None or len(allowed_vec_ids) == 0:
            return []

    D, I = _faiss_search(faiss_index, q_vec, fetch_k, allowed_vec_ids=allowed_vec_ids)

    raw_results: list[dict] = []
    seen_vecs: set[int] = set()
    for dist, vec_id in zip(D[0], I[0]):
        vec_id = int(vec_id)
        if vec_id < 0 or vec_id in seen_vecs:
            continue
        seen_vecs.add(vec_id)

        result = _make_result_from_hit(
            query=query,
            faiss_index=faiss_index,
            dist=float(dist),
            vec_id=vec_id,
            mapping_df=mapping_df,
            pages_df=pages_df,
            alpha=alpha,
            max_snippet_length=max_snippet_length,
            year_filtered=bool(year is not None and year2vec),
        )
        if not result:
            continue
        if float(result.get("score", 0.0)) < float(threshold):
            continue
        raw_results.append(result)

    if not raw_results:
        return []

    raw_results.sort(key=lambda x: x.get("weighted_score", 0.0), reverse=True)
    faiss_docs = merge_snippets(raw_results, max_snippets_per_doc=int(_cfg("MAX_SNIPPETS_PER_DOC", 4)))

    # Hybrid: fuse FAISS + BM25 with RRF. Skip when year-filtering since BM25 has no year filter.
    if _bm25_cache is not None and year is None:
        bm25_docs = _bm25_search(query, pages_df, k=int(k) * 2, alpha=alpha, max_snippet_length=max_snippet_length)
        if bm25_docs:
            return _fuse_ranked_results(query, [faiss_docs, bm25_docs], k=int(k))

    return _rerank_results(query, faiss_docs, limit=int(k))


# ---------------------------------------------------------------------------
# Search caching and multi-query fusion
# ---------------------------------------------------------------------------

def _normalise_cache_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return json.dumps([_normalise_cache_value(x) for x in v], sort_keys=True)
    return str(v)


def make_search_cache_key(
    query: str,
    k: int,
    alpha: float,
    max_snippet_length: int,
    threshold: float,
    year: int | None,
) -> str:
    raw = "|".join(
        [
            SEARCH_VERSION,
            _normalise_cache_value(config.EMBEDDING_MODEL),
            _normalise_cache_value(query),
            _normalise_cache_value(k),
            _normalise_cache_value(alpha),
            _normalise_cache_value(max_snippet_length),
            _normalise_cache_value(threshold),
            _normalise_cache_value(year),
            _normalise_cache_value(_cfg("RERANKER_MODEL", None)),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached_search_results(cache_key: str) -> list[dict] | None:
    return st.session_state.get("search_cache", {}).get(cache_key)


def store_search_results(cache_key: str, results: list[dict]):
    cached = st.session_state.get("search_cache", {})
    cached[cache_key] = results
    st.session_state["search_cache"] = cached


def search_pdfs_cached(
    query: str,
    faiss_index,
    embeddings,
    mapping_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    *,
    k: int = config.SEARCH_RESULT_K,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    threshold: float = 0.0,
    year2vec: dict[int, np.ndarray] | None = None,
    year: int | None = None,
) -> list[dict]:
    query = clean_text(query)
    if not query:
        return []

    cache_key = make_search_cache_key(query, k, alpha, max_snippet_length, threshold, year)
    cached = cached_search_results(cache_key)
    if cached is not None:
        return cached

    results = search_pdfs(
        query,
        faiss_index,
        embeddings,
        mapping_df,
        pages_df,
        k=k,
        alpha=alpha,
        max_snippet_length=max_snippet_length,
        threshold=threshold,
        year2vec=year2vec,
        year=year,
    )
    store_search_results(cache_key, results)
    return results


def _dedupe_queries(queries: Iterable[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        q = clean_text(q)
        if not q:
            continue
        norm = re.sub(r"\W+", " ", q.lower()).strip()
        if not norm or norm in seen:
            continue
        out.append(q)
        seen.add(norm)
        if len(out) >= limit:
            break
    return out


def _rule_based_query_variants(query: str) -> list[str]:
    variants = [query]
    terms = _tokenize(query)
    expanded = query
    for term in terms:
        expansion = _ACRONYM_EXPANSIONS.get(term.lower())
        if expansion and expansion.lower() not in expanded.lower():
            expanded = f"{expanded} {expansion}"
    if expanded != query:
        variants.append(expanded)
    return variants




def make_multiquery_cache_key(
    queries: Sequence[str],
    k: int,
    alpha: float,
    max_snippet_length: int,
    threshold: float,
    year: int | None,
) -> str:
    raw = "|".join(
        [
            SEARCH_VERSION,
            "multi",
            json.dumps(list(queries), sort_keys=True),
            _normalise_cache_value(k),
            _normalise_cache_value(alpha),
            _normalise_cache_value(max_snippet_length),
            _normalise_cache_value(threshold),
            _normalise_cache_value(year),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fuse_ranked_results(query: str, ranked_lists: Sequence[list[dict]], *, k: int) -> list[dict]:
    fused: dict[str, dict] = {}

    for list_idx, hits in enumerate(ranked_lists):
        list_weight = 1.0 if list_idx == 0 else 0.85
        for rank, hit in enumerate(hits, start=1):
            key = _doc_key(hit)
            score = list_weight / (RRF_K + rank)
            if key not in fused:
                item = hit.copy()
                item["rrf_score"] = 0.0
                item["query_match_count"] = 0
                item["_snippets"] = []
                fused[key] = item
            item = fused[key]
            item["rrf_score"] += score
            item["query_match_count"] += 1
            item["_snippets"].append(hit.get("snippet", ""))
            # Preserve the best metadata/scoring fields.
            if hit.get("combined_score", 0.0) > item.get("combined_score", 0.0):
                for field, value in hit.items():
                    if not field.startswith("_"):
                        item[field] = value

    results: list[dict] = []
    for item in fused.values():
        snippets: list[str] = []
        seen: set[str] = set()
        for snip in item.pop("_snippets", []):
            snip_clean = clean_text(snip)
            key = snip_clean.lower()[:240]
            if snip_clean and key not in seen:
                snippets.append(snip_clean)
                seen.add(key)
            if len(snippets) >= int(_cfg("MAX_SNIPPETS_PER_DOC", 4)):
                break
        if snippets:
            item["snippet"] = " ... ".join(snippets)

        item["combined_score"] = (
            float(item.get("combined_score", item.get("weighted_score", 0.0)))
            + 4.0 * float(item.get("rrf_score", 0.0))
            + 0.03 * min(3, int(item.get("query_match_count", 1)))
        )
        results.append(item)

    results.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return _rerank_results(query, results, limit=k)


def search_pdfs_multiquery_cached(
    queries: Sequence[str] | str,
    faiss_index,
    embeddings,
    mapping_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    *,
    k: int = config.SEARCH_RESULT_K,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    threshold: float = 0.0,
    year2vec: dict[int, np.ndarray] | None = None,
    year: int | None = None,
) -> list[dict]:
    if isinstance(queries, str):
        queries = [queries]

    expanded: list[str] = []
    for q in queries:
        expanded.extend(_rule_based_query_variants(q))
    queries = _dedupe_queries(expanded, limit=max(1, DEFAULT_MAX_SEARCH_QUERIES))
    if not queries:
        return []

    cache_key = make_multiquery_cache_key(queries, k, alpha, max_snippet_length, threshold, year)
    cached = cached_search_results(cache_key)
    if cached is not None:
        return cached

    per_query_k = max(int(k), int(k) * 2)
    ranked_lists = [
        search_pdfs_cached(
            q,
            faiss_index,
            embeddings,
            mapping_df,
            pages_df,
            k=per_query_k,
            alpha=alpha,
            max_snippet_length=max_snippet_length,
            threshold=threshold,
            year2vec=year2vec,
            year=year,
        )
        for q in queries
    ]
    results = _fuse_ranked_results(queries[0], ranked_lists, k=int(k))
    store_search_results(cache_key, results)
    return results


# ---------------------------------------------------------------------------
# Prompt/context construction and citation streaming
# ---------------------------------------------------------------------------

def _history_to_messages(history: List[dict], *, limit: int | None = None) -> list:
    recent = history[-limit:] if limit else history
    messages = []
    for m in recent:
        role = str(m.get("role", "")).lower()
        content = str(m.get("content", ""))
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
        elif role == "system":
            continue
    return messages

def build_context_blocks(docs: list[dict], *, max_chars: int = DEFAULT_CONTEXT_MAX_CHARS) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    kept: list[dict] = []
    used_chars = 0

    for d in docs:
        title = clean_text(str(d.get("title") or "Untitled"), max_chars=220)
        url = d.get("url") or d.get("pdf_url") or "#"
        date = d.get("publication_date") or "unknown date"
        pub_type = d.get("publication_type") or "unknown type"
        snippet = clean_text(d.get("snippet", ""), max_chars=int(_cfg("CONTEXT_SNIPPET_MAX_CHARS", 1_100)))
        summary = clean_text(str(d.get("summary") or ""), max_chars=350)

        if not snippet:
            continue

        idx = len(kept) + 1  # always consecutive — no gaps even when docs are skipped
        bits = [f"[{idx}] {title}", f"Date: {date}", f"Type: {pub_type}", f"URL: {url}"]
        if summary:
            bits.append(f"Summary: {summary}")
        bits.append(f"Evidence: {snippet}")
        block = "\n".join(bits)

        if used_chars + len(block) > max_chars and kept:
            break
        blocks.append(block)
        item = d.copy()
        item["ref"] = str(idx)
        kept.append(item)
        used_chars += len(block)

    return "\n\n".join(blocks), kept


def _citation_regex() -> re.Pattern:
    return re.compile(
        r"(?<!\[\^)"                  # ignore already-converted footnotes
        r"[\[(]"                       # opening [ or (
        r"("                            # capture id list
        r"(?:\d{1,3}|\d{4}-\d+)"      # plain source number or YYYY-n style
        r"(?:\s*,\s*(?:\d{1,3}|\d{4}-\d+))*"
        r")"
        r"[\])]"                       # closing ] or )
    )


def yield_answer_with_citations(
    model,
    messages,
    docs: List[Dict[str, Any]],
    *,
    emit_sources: bool | None = None,
) -> Iterator[str]:
    citation_rx = _citation_regex()
    adjacent_fn_rx = re.compile(r"(\[\^[^\]]+\])(\s*)(?=\[\^[^\]]+\])")

    citations: Dict[str, Dict[str, Any]] = {str(i): d for i, d in enumerate(docs, 1)}
    for d in docs:
        if d.get("ref"):
            citations[str(d["ref"])] = d

    used_refs: Set[str] = set()
    buffer = ""
    fn_ref_rx = re.compile(r"\[\^(\w+)\]")

    def _repl(match: re.Match) -> str:
        ids = [i.strip() for i in match.group(1).split(",")]
        valid_ids = [i for i in ids if i in citations]
        used_refs.update(valid_ids)
        if not valid_ids:
            return match.group(0)
        return " ".join(f"[^{i}]" for i in valid_ids)

    def _flush(text: str) -> str:
        out = citation_rx.sub(_repl, text)
        out = adjacent_fn_rx.sub(lambda m: f"{m.group(1)}, ", out)
        # Track [^N] refs whether they came from _repl or were output directly by the model.
        for m in fn_ref_rx.finditer(out):
            if m.group(1) in citations:
                used_refs.add(m.group(1))
        return out

    for chunk in model.stream(messages):
        token = chunk.content or ""
        buffer += token
        if len(buffer) > 600 or any(sep in buffer for sep in (". ", "\n")):
            yield _flush(buffer)
            buffer = ""

    if buffer:
        yield _flush(buffer)

    if emit_sources is None:
        emit_sources = bool(docs)
    if not emit_sources or not docs:
        return

    yield "\n\n---\n\n"

    refs_to_show = sorted(used_refs, key=lambda r: int(r.split("-")[-1]) if r.split("-")[-1].isdigit() else r)
    if not refs_to_show:
        refs_to_show = [str(i) for i in range(1, min(4, len(docs)) + 1)]
        yield "*The answer did not emit inline citations, but these were the retrieved sources consulted.*\n\n"

    for ref in refs_to_show:
        doc = citations.get(ref)
        if not doc:
            continue
        link = doc.get("url") or doc.get("pdf_url") or "#"
        title = clean_text(str(doc.get("title") or "Untitled"), max_chars=220)
        pub_date = doc.get("publication_date") or ""
        pub_type = doc.get("publication_type") or "Unknown"
        snippet = clean_text(doc.get("snippet") or doc.get("summary") or "", max_chars=500)
        excerpt = f"\n*{snippet}*" if snippet else ""
        yield f"[^{ref}]: {pub_date} - [{title}]({link}) - {pub_type}.{excerpt}\n"


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

def chat_rag(
    prompt: str,
    history: List[dict],
    *,
    faiss_index,
    embeddings,
    mapping_df: pd.DataFrame,
    pages_df: pd.DataFrame,
    k: int = 5,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    callbacks: list | None = None,
    openai_api_key: str | None = None,
    llm_model: str = "gpt-4o-mini",
) -> Generator[str, None, None]:
    """Always-RAG streaming generator: retrieves, builds context, streams cited answer."""
    search_queries = _dedupe_queries(
        _rule_based_query_variants(clean_text(prompt)),
        limit=DEFAULT_MAX_SEARCH_QUERIES,
    )

    docs = search_pdfs_multiquery_cached(
        search_queries,
        faiss_index,
        embeddings,
        mapping_df,
        pages_df,
        k=int(k),
        alpha=alpha,
        max_snippet_length=max_snippet_length,
        threshold=float(_cfg("SIMILARITY_THRESHOLD", 0.0)),
    )

    context, docs = build_context_blocks(docs, max_chars=DEFAULT_CONTEXT_MAX_CHARS)
    if not context:
        yield (
            "I searched the Transport & Environment / Clean Cities Campaign document index, "
            "but did not find a relevant source. Try a more specific term, report title, "
            "policy name, year, or acronym."
        )
        return

    sys_ctx = SystemMessage(
        content=(
            "You are a meticulous research assistant for Transport & Environment. "
            "Retrieved context is the primary source of truth. Answer only from the retrieved "
            "context. Cite every factual paragraph with bracketed source numbers like [1]. "
            "Do not cite sources that do not support the sentence. If the context is thin, "
            "say what is missing instead of filling gaps from memory. Be direct."
        )
    )
    messages = [
        sys_ctx,
        SystemMessage(content=f"Retrieved context documents:\n\n{context}"),
        *_history_to_messages(history, limit=DEFAULT_HISTORY_TURNS),
        HumanMessage(content=prompt),
    ]

    model = ChatOpenAI(
        model_name=llm_model,
        streaming=True,
        callbacks=callbacks,
        openai_api_key=openai_api_key,
        temperature=float(_cfg("ANSWER_TEMPERATURE", 0.0)),
    )

    yield from yield_answer_with_citations(model, messages, docs, emit_sources=bool(docs))


# ---------------------------------------------------------------------------
# Timeline utility
# ---------------------------------------------------------------------------

def _parse_json_list(text: str) -> list[int]:
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\[[\s\S]*\]", text or "")
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def position_timeline(
    topic: str,
    *,
    faiss_index,
    embeddings,
    mapping_df,
    pages_df,
    year2vec,
    openai_api_key: str,
    alpha: float = 0.0,
    max_snippet_length: int = 500,
    k_per_year: int = config.TOP_HITS_PER_YEAR,
    min_score: float = config.SIMILARITY_THRESHOLD,
    triage_model: str = config.TRIAGE_MODEL,
    timeline_model: str = config.TIMELINE_MODEL,
):
    """
    Stream a Markdown timeline answering:
    "What is T&E's position on <topic> and how did it change?"
    """
    topic = clean_text(topic)
    if not topic:
        yield "Please provide a topic."
        return

    now_year = datetime.now().year
    years = [int(y) for y in sorted(year2vec) if int(y) <= now_year]
    all_hits: list[dict] = []

    for yr in sorted(years, reverse=True):
        hits = search_pdfs_cached(
            topic,
            faiss_index=faiss_index,
            embeddings=embeddings,
            mapping_df=mapping_df,
            pages_df=pages_df,
            k=k_per_year,
            alpha=alpha,
            max_snippet_length=max_snippet_length,
            threshold=min_score,
            year2vec=year2vec,
            year=yr,
        )
        for h in hits:
            h["year"] = yr
        all_hits.extend(hits)

    if not all_hits:
        yield "No documents above the similarity threshold were found for this topic."
        return

    triage_chunks = []
    for idx, h in enumerate(all_hits, start=1):
        triage_chunks.append(f"[{idx}] ({h['year']}) {h.get('title') or 'Untitled'}\n{h.get('snippet') or ''}")

    triage_prompt = (
        "You are a policy analyst at Transport & Environment/Clean Cities Campaign. "
        "Given the snippets below, select only those that are relevant to the organisation's "
        f"own position or stance on: {topic}. Return a JSON list of reference numbers.\n\n"
        "Snippets:\n" + "\n\n".join(triage_chunks)
    )

    triager = ChatOpenAI(
        model_name=triage_model,
        openai_api_key=openai_api_key,
        temperature=0,
    )
    try:
        triage_reply = triager.invoke([HumanMessage(content=triage_prompt)]).content
        keep_ids = set(_parse_json_list(triage_reply))
    except Exception:
        keep_ids = set(range(1, len(all_hits) + 1))

    kept_hits = [h for i, h in enumerate(all_hits, start=1) if i in keep_ids]
    if not kept_hits:
        yield "No publication explicitly states T&E's position on this topic."
        return

    kept_hits.sort(key=lambda x: (-int(x.get("year", 0)), -float(x.get("weighted_score", 0.0))))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for h in kept_hits:
        yr = int(h.get("year", 0))
        if len(grouped[yr]) < k_per_year:
            grouped[yr].append(h)

    docs = []
    ctx_blocks = []
    global_idx = 1
    for yr in sorted(grouped):  # chronology: oldest to newest
        for h in grouped[yr]:
            ref = str(global_idx)
            snippet = clean_text(h.get("snippet", ""), max_chars=900)
            title = clean_text(str(h.get("title") or "Untitled"), max_chars=220)
            ctx_blocks.append(f"[{ref}] ({yr}) {title}\n{snippet}")
            item = h.copy()
            item["ref"] = ref
            docs.append(item)
            global_idx += 1

    sys_ctx = SystemMessage(
        content=(
            "You are an expert policy summariser at Transport & Environment. Using only the "
            "provided snippets, compose a chronological timeline from earlier to later explaining "
            "how T&E's position evolved. For each year, write 1-3 concise bullets. Cite each "
            "position claim with bracketed source numbers like [1]. If the evidence does not show "
            "a change, say that clearly. Finish with a two-sentence summary.\n\n"
            "Snippets:\n" + "\n\n".join(ctx_blocks)
        )
    )

    model = ChatOpenAI(
        model_name=timeline_model,
        openai_api_key=openai_api_key,
        streaming=True,
    )

    yield from yield_answer_with_citations(model, [sys_ctx], docs, emit_sources=True)


def make_timeline_cache_key(
    topic: str,
    alpha: float,
    max_snippet_length: int,
    k_per_year: int,
    min_score: float,
    triage_model: str,
    timeline_model: str,
) -> str:
    raw = "|".join(
        [
            SEARCH_VERSION,
            topic,
            _normalise_cache_value(alpha),
            _normalise_cache_value(max_snippet_length),
            _normalise_cache_value(k_per_year),
            _normalise_cache_value(min_score),
            triage_model,
            timeline_model,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_timeline(cache_key: str) -> str | None:
    return st.session_state.get("timeline_cache", {}).get(cache_key)


def store_timeline(cache_key: str, timeline: str):
    cached = st.session_state.get("timeline_cache", {})
    cached[cache_key] = timeline
    st.session_state["timeline_cache"] = cached
