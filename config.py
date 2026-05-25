### **`config.py`**
import os
# Embedding Model
EMBEDDING_MODEL      = "intfloat/multilingual-e5-small"
SEARCH_RESULT_K      = 5

DEFAULT_MODEL           = "gpt-5.4-mini"
TRIAGE_MODEL            = "gpt-5.4-mini"
TIMELINE_MODEL          = "gpt-5.4-mini"

TOP_HITS_PER_YEAR       = 3   
SIMILARITY_THRESHOLD    = 0.75

# Where you stored the artefacts when building the index
INDEX_PATH   = "./embeddings/multilingual-e5-small-docs.index"               # faiss.write_index(...)
MAP_PATH     = "./embeddings/multilingual-e5-small-faiss_mapping.parquet"    # vector_id → doc_id / offsets
PAGES_PATH   = "./embeddings/metadata_with_fulltext.parquet"                 # fulltext + metadata
BM25_PATH    = "./embeddings/bm25_index.pkl"                                 # pre-built BM25 (pickle)

HF_CACHE_DIR = "./hf_model_cache"

os.environ["HF_HOME"] = HF_CACHE_DIR

# Streamlit App
LOGO_PATH = "assets/logo.png"     # Path to your company logo
APP_TITLE = "Internal Knowledge Database"  # Title for the Streamlit app

# Other Settings
DEFAULT_QUERY = "Type your search query here..."  # Placeholder text for the search bar

ENABLE_MULTI_QUERY = True
MAX_SEARCH_QUERIES = 4
SEARCH_FETCH_MULTIPLIER = 8
CONTEXT_MAX_CHARS = 14000
MAX_SNIPPETS_PER_DOC = 4
CONTEXT_SNIPPET_MAX_CHARS = 1100

# Optional but recommended if you can afford local reranking latency:
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Important:
# If your index was built with E5-style query/document prefixes, keep this:
EMBEDDING_QUERY_PREFIX = "query: "
#
# For OpenAI embeddings, leave this as None or omit it.
#EMBEDDING_QUERY_PREFIX = None

ANSWER_TEMPERATURE = 0.0
SEARCH_VERSION = "rag-v2.1"