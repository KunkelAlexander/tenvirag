"""
bugreport.py
-------------
Create GitHub issues directly from a Streamlit app.

Required secrets:
- GITHUB_TOKEN : GitHub fine-grained PAT with Issues: RW
- GITHUB_REPO  : "owner/repo"
"""

from __future__ import annotations

import streamlit as st
import requests
import json
import datetime
import traceback
from typing import Any


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

REDACT_KEYS = {
    "API_KEY",
    "OPENAI_API_KEY",
    "password",
    "PASSWORD",
    "token",
    "TOKEN",
    "secret",
    "SECRET",
}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _redact(obj: Any) -> Any:
    """
    Recursively redact sensitive values from dicts / lists.
    """
    if isinstance(obj, dict):
        return {
            k: ("***REDACTED***" if any(x in k.upper() for x in REDACT_KEYS) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj

def _safe_session_state() -> dict:
    """
    Capture a filtered, JSON-safe snapshot of session state.
    """

    SAFE_KEYS = {
        # navigation
        "active_tab",

        # search / chat inputs
        "search_query",
        "chrono_query",
        "position_query",
        "chat_history",

        # config / sliders
        "alpha",
        "max_snippet_length",
        "n_search_results",
        "n_chrono_search_results",
        "chrono_similarity_threshold",
        "position_similarity",
        "position_hits",
        "selected_model",
    }

    snapshot = {}

    for key in SAFE_KEYS:
        if key in st.session_state:
            try:
                snapshot[key] = _redact(st.session_state[key])
            except Exception:
                snapshot[key] = "<unserialisable>"

    return snapshot



# ──────────────────────────────────────────────────────────────
# Issue body builder
# ──────────────────────────────────────────────────────────────
def build_issue_body(
    user_description: str | None = None,
    exception: Exception | None = None,
    include_state: bool = True,
) -> str:
    sections = []

    sections.append("### 🐞 Bug description")
    sections.append(user_description or "_No user description provided_")

    if exception:
        sections.append("\n---\n### 💥 Exception")
        sections.append("```")
        sections.append("".join(traceback.format_exception(exception)))
        sections.append("```")

    if include_state:
        sections.append("\n---\n### 🧠 Session state")
        sections.append("```json")
        sections.append(
            json.dumps(
                _safe_session_state(),
                indent=2,
                default=str,
            )
        )
        sections.append("```")

    sections.append("\n---\n### ⏱ Metadata")
    sections.append(f"- Timestamp: {datetime.datetime.utcnow().isoformat()} UTC")
    sections.append(f"- Streamlit version: {st.__version__}")

    return "\n".join(sections)



# ──────────────────────────────────────────────────────────────
# GitHub API
# ──────────────────────────────────────────────────────────────

def open_github_issue(
    title: str,
    body: str,
    labels: list[str] | None = None,
):
    """
    Open a GitHub issue.

    Raises if the request fails.
    """

    repo = st.secrets["GITHUB_REPO"]
    token = st.secrets["GITHUB_TOKEN"]

    url = f"https://api.github.com/repos/{repo}/issues"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    payload = {
        "title": title,
        "body": body,
        "labels": labels or ["bug"],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=10)
    r.raise_for_status()


# ──────────────────────────────────────────────────────────────
# High-level convenience APIs
# ──────────────────────────────────────────────────────────────

def report_user_bug(
    description: str,
    include_state: bool = True,
    title: str = "Bug report from Streamlit app",
    labels: list[str] | None = None,
):
    body = build_issue_body(
        user_description=description,
        include_state=include_state,
    )
    open_github_issue(title=title, body=body, labels=labels)

def report_exception(
    exception: Exception,
    title_prefix: str = "Crash",
    labels: list[str] | None = None,
):
    """
    Call this from except blocks.
    Never crashes the app if reporting fails.
    """
    try:
        title = f"{title_prefix}: {type(exception).__name__}"
        body = build_issue_body(exception=exception)
        open_github_issue(
            title=title,
            body=body,
            labels=labels or ["bug", "crash"],
        )
    except Exception:
        # Never let bug reporting crash the app
        pass
