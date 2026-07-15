# TenviRAG

TenviRAG is a lightweight research assistant for exploring and analysing Transport & Environment publications.
It combines a FAISS vector index with a Streamlit interface that supports semantic search, retrieval‑augmented chat, and a set of more specialised exploration modes.

<p align="center">
  <img src="figures/user_experience.gif" alt="Demo animation" width="700">
</p>


## Overview

The application is designed to make a large corpus of policy and research documents easier to navigate. Users can:

* search across publications using semantic similarity rather than keywords,
* ask conversational questions grounded in the underlying documents,
* explore how topics and positions evolve over time,
* report issues directly from the interface.

The system is intentionally modular and relatively small, so it can be adapted to new document sources, models, or ranking strategies.

## Main features

### Semantic search

* Sentence‑Transformer embeddings (`multilingual-e5-small`) with multilingual support
* FAISS HNSW index for fast retrieval
* Optional time‑based decay so more recent publications can be prioritised
* Configurable number of results and snippet length

### Chat and RAG

* Conversational interface backed by retrieval‑augmented generation
* The model only queries the document index when needed
* Answers include inline citations and a list of sources
* Configurable OpenAI model via the sidebar

### Chronological exploration

* Dedicated “Chronological” view to inspect results year by year
* Similarity thresholds to filter weak matches
* Useful for tracking how a topic appears over time

### Position timelines

* A “Position” view that reconstructs how T&E positions on a topic evolve
* Year‑aware retrieval with per‑year limits and minimum similarity thresholds
* Cached timelines to avoid recomputation

### Streamlit interface

* Tab‑based layout (Chat, Search, Chronological, Position)
* Sidebar with expert settings (ranking, decay, snippet length, model choice)
* API key or password‑based access via the sidebar
* Reset button to clear chat history without restarting the app

### Bug reporting

* Built‑in bug report dialog (implemented in `bugreport.py`)
* Users can submit issues directly from the UI
* Reports are forwarded to GitHub Issues with relevant context

## Installation

### Prerequisites

* Python 3.8 or newer
* `pip`
* Optional: `virtualenv` or `conda`

### Clone and set up

- Requires git lfs for the large index that exceeds 100MB

```bash
git clone https://github.com/KunkelAlexander/t-e_search_tool.git
cd t-e_search_tool
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.py` to adjust:

* paths and cache locations,
* embedding and language models,
* ranking and timeline parameters.

At runtime, provide an OpenAI API key (or application password) via the Streamlit sidebar.


## Usage

### Build or update the index

The notebook below handles PDF ingestion, chunking, embedding, and FAISS indexing:

```bash
build_index.ipynb
```

### Launch the application

```bash
streamlit run frontend/app.py
```

### Search

Use the **Search** tab to enter a query and inspect ranked results. Each result shows publication metadata, a similarity score, and a highlighted snippet.

### Chat

Use the **Chat** tab to ask questions in natural language. Answers are grounded in the document corpus and include citations.

### Chronological and position views

Use the **Chronological** and **Position** tabs to explore how topics or policy positions develop over time, with fine‑grained control over similarity and result limits.

### Reporting a bug

Use the in‑app bug report option to submit feedback or errors. Reports are automatically filed as GitHub issues for tracking and follow‑up.


## Project structure

```
frontend/app.py      Streamlit application
search.py            Retrieval, ranking, and RAG logic
bugreport.py         In‑app bug reporting to GitHub Issues
embeddings/          Cached indices and metadata
figures/             Screenshots
assets/              Demo assets
```


## Roadmap

* Scheduled ingestion of new publications
* Additional evaluation and monitoring for retrieval quality


## License

MIT License (see `LICENSE`).

## Disclaimer

This is a personal side project by Alexander Kunkel.
It is not an official product of Transport & Environment and does not represent the organisation’s views or policies.
Use at your own discretion.
