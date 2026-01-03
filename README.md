# TenviRAG


A lightweight **research‑assistant platform** that lets you _chat_ with T&E's publication built with the help of LLMs.
It couples a **FAISS** vector store with a **Streamlit** UI that offers both RAG-powered search and an AI‑powered chat interface.

<p align="center">
  <img src="figures/user_experience.gif" alt="Demo animation" width="700">
</p>

---

## ✨ Key Features

| Area | Highlights |
|------|------------|
| **Semantic Search** | • Sentence‑Transformer embeddings ('multilingual-e5-small') with multi-language support <br>• Optional exponential **date‑decay** weighting so fresh material floats to the top |
| **Storage** | • FAISS HNSW index for millisecond retrieval<br>• All vectors + metadata **persisted** on disk |
| **Streamlit Front‑End** | • Responsive two‑tab layout – **Search** & **Chat**<br>• Clickable results with similarity colouring<br>• Floating chat bar, expert settings sliders |
| **Retrieval‑Augmented Chat (RAG)** | • Router decides when to query the corpus<br>• Sources block with inline `[1]` citations |
| **Extensibility** | Simple, modular Python; swap embedding models, adjust ranking, plug‑in new data loaders |

---

## 🛠 Installation

### 1. Prerequisites
* Python ≥ 3.8
* `pip` package manager
* (optional) `virtualenv` or `conda`

### 2. Clone & set up
```bash
git clone https://github.com/KunkelAlexander/t-e_search_tool.git
cd t-e_search_tool
python -m venv venv                 # optional but recommended
source venv/bin/activate            # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure
Edit **`config.py`** (paths, model names, index size, etc.).
At runtime, supply your **OpenAI key** via:

* the Streamlit sidebar input

---

## 🚀 Usage

### 1. Build / update the index
```bash
build_index.ipynb   # scrapes PDFs → embeddings → FAISS
```
Adjust model, chunk size and filters in `config.py`.

### 2. Launch the UI
```bash
streamlit run frontend/app.py
```

### 3. Search
*Switch to the **Search** tab, type a query.*
Results show publication type, date and a colour‑coded match score.

### 4. Chat
Ask conversational questions in the **Chat** tab.
The assistant will cite snippets (`[1]`) and list full sources below its answer.

---

## ⚙ Expert Settings (in the sidebar)

| Control | Effect |
|---------|--------|
| **# Search Results** | top‑*k* candidates returned from FAISS |
| **Date Decay α** | how strongly older docs are down‑weighted |
| **Max Snippet Length** | truncate long excerpts for brevity |

---

## 🧑‍💻 Project Structure

```
app.py              main entry point
search.py           implement rag retrieval and chat using langchain
build_index.ipynb   create faiss vector database and read pdfs
embeddings/         cached artefacts (index, parquet mapping,…)
figures/            screenshots / GIFs
```

## 🌱 Roadmap

* Faceted filters (author, year, tag) in the UI
* Scheduled crawler to auto‑ingest new publications

---

## 📝 License
[MIT](LICENSE)

---

## 🔖 Disclaimer
This is a personal side‑project by **Alexander Kunkel**.
It is **not** an official product of **Transport & Environment** and reflects only the author’s views.
Use at your own discretion.
