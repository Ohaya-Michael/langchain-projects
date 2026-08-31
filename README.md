<h1 align="center">🏙️🩺 Two-Domain RAG Pipeline</h1>

<p align="center">
  <em>One retrieval-augmented system that answers questions across two unrelated worlds —
  Berlin real-estate listings and medical claims — then grades and monitors its own answers.</em>
</p>

<p align="center">
  <img alt="Python"     src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="LangChain"  src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white">
  <img alt="LangGraph"  src="https://img.shields.io/badge/LangGraph-StateGraph-FF6F00">
  <img alt="Chroma"     src="https://img.shields.io/badge/Chroma-VectorDB-5A2FD6">
  <img alt="Streamlit"  src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="OpenAI"     src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white">
</p>

<p align="center">
  <a href="https://github.com/Ohaya-Michael/langchain-projects">
    <img alt="Repo" src="https://img.shields.io/badge/GitHub-Ohaya--Michael%2Flangchain--projects-181717?logo=github&logoColor=white">
  </a>
</p>

<p align="center">
  📦 <strong>Repository:</strong> <a href="https://github.com/Ohaya-Michael/langchain-projects">github.com/Ohaya-Michael/langchain-projects</a>
</p>

---

## ✨ Overview

Instead of letting a language model answer from memory, this project **retrieves real
documents first** and makes the model answer only from those — with sources cited. Around
that core it adds the parts that turn a demo into something trustworthy: query rewriting,
automatic routing between two knowledge bases, relevance grading, an **LLM-as-judge
evaluator**, per-query logging, and a **live monitoring dashboard**.

The whole flow is built as a `langgraph` `StateGraph` in `using_langchain_langgraph.py`,
wrapped as an API in `langchain_chain_rag_function_api.py`, and observed through
`dashboard.py`.

## 🚀 Features

- 🧭 **Smart routing** — each question is classified as *real-estate* or *medical* and only the right store is searched.
- ✍️ **Query rewriting** — fixes spelling and expands abbreviations (`COPD → COPD (chronic obstructive pulmonary disease)`) before retrieval.
- 🧹 **Relevance grading** — a binary grader drops off-topic documents so answers stay grounded.
- 📝 **Sourced synthesis** — every fact is tagged with its origin: `(Real Estate)`, `(Synthea)`, `(CMS SynPUF)`.
- ⚖️ **Self-evaluation** — an LLM-as-judge scores each answer for faithfulness, relevance, and hallucination.
- 💾 **Full logging** — every run is saved to `query_results/` as JSON — nothing is a black box.
- 📊 **Monitoring dashboard** — a Streamlit + pandas app turns those logs into live quality metrics.
- 🔌 **API-ready** — the pipeline is exposed behind an endpoint for integration into other apps.

## 🧠 How it works

The pipeline is a `StateGraph` over a shared `RAGState`. Each node returns a partial dict
that merges into the state; one conditional edge picks the retriever:

```
START → rewrite → route ─┬─→ retrieve_real_estate ─┐
                         └─→ retrieve_medical ──────┴─→ grade → generate → evaluate → save → END
```

| Node | What it does |
|------|--------------|
| **rewrite** | Turns the raw question into a clean, self-contained search query |
| **route** | Classifies the query as `medical` or `real_estate` (structured output) |
| **retrieve** | Runs only the matching retriever (`k=5`) |
| **grade** | Keeps only documents relevant to the question |
| **generate** | Synthesizes one sourced answer from the graded context |
| **evaluate** | Scores the answer with an LLM-as-judge |
| **save** | Writes the full record to `query_results/<timestamp>_<slug>.json` |

The LLM throughout is `gpt-4o-mini` at `temperature=0`.

## 🗂️ The stores

| Store | Collection | Persist dir | What it holds |
|-------|-----------|-------------|---------------|
| `real_estate` | `immobilien` | `./chroma_immobilien` | Berlin resale, rental, new-build listings, monthly Kiez stats, transit stations |
| `medical` | `medicine_claims` | `./chroma_medicine` | `synthea` patient records + `cms_synpuf` 2008–2010 Medicare claims |

Both are embedded locally with `sentence-transformers`. The `data_source` metadata field
(`real_estate`, `synthea`, `cms_synpuf`) is what lets each answer cite where its facts came from.

> 🧩 **Design choice:** the knowledge bases can live as **separate vector stores per domain**,
> *or* as a **single shared store** where each document carries a `data_source` metadata tag
> and is filtered at query time. This project supports both — so adding a third or fourth
> source later needs no redesign.

## ⚖️ RAG evaluation (LLM-as-judge)

Every answer is scored automatically by a second model acting as a strict judge, using only
the question, the retrieved context, and the generated answer. The scores are stored with
each run so quality is measurable, not vibes-based.

| Field | Range | Meaning |
|-------|-------|---------|
| `faithfulness` | 1–5 | Is every claim grounded in the retrieved context? |
| `answer_relevance` | 1–5 | How directly does the answer address the question? |
| `context_relevance` | 1–5 | How relevant were the retrieved documents? |
| `hallucinated` | true/false | Does the answer state facts not present in the context? |
| `rationale` | text | One or two sentences justifying the scores |

Because the judge is grounded in the retrieved context, it penalizes any claim the sources
don't support — the same discipline the synthesis prompt enforces. Scores land in each
`query_results/*.json` record, ready for the dashboard below.

## 📊 Monitoring dashboard (Streamlit + pandas)

`dashboard.py` reads the `query_results/*.json` logs with **pandas** and renders a live
monitoring view with **Streamlit** — no database required, it just watches the folder.

```bash
pip install streamlit pandas
streamlit run dashboard.py
# point at a different folder:
streamlit run dashboard.py -- --log-dir ./query_results
```

What it surfaces:

- 📈 **KPI row** — total queries, average faithfulness / answer-relevance / context-relevance, hallucination rate, and % of "no-context" answers.
- 🕒 **Scores over time** — daily average of each judge metric, so regressions are obvious.
- 🥧 **Datasource split** and **hallucinations by datasource** — see whether medical or real-estate queries are the weak spot.
- 📊 **Score distributions** — 1–5 histograms per metric.
- 🔎 **Drill-down** — inspect any run: rewritten query, answer, judge rationale, and the retrieved documents with their source tags.
- 🎛️ **Filters** — log directory, datasource, date range, and an "only hallucinated" toggle.

The dashboard is fully decoupled from the pipeline: it needs only `dashboard.py` and the
`query_results/` folder, and tolerates missing fields or malformed files without crashing.

## 📁 Project layout

```
langchain_project/
├── README.md                             # this file
├── README_vectordb.md                    # how the immobilien vector DB is built
├── requirements.txt                      # Python dependencies
├── __init__.py
├── build_immobilien_vectordb.py          # Berlin CSVs  -> chroma_immobilien
├── query_chroma_db_vectordb.py           # CLI to query a Chroma store directly
├── using_langchain_langgraph.py          # the LangGraph RAG pipeline
├── using_langchain_langgraph.ipynb       # notebook version of the pipeline
├── langchain_chain_rag_function_api.py   # chain backend (local Chroma), FastAPI `app`
├── using_langchain_langgraph_cloud.py    # LangGraph backend over Qdrant Cloud (build_pipeline)
├── langgraph_api.py                      # FastAPI wrapper -> serves the LangGraph backend
├── rag_streamlit_app.py                  # Streamlit UI — HTTP client of the RAG API
├── dashboard.py                          # Streamlit + pandas monitoring dashboard
├── Dockerfile.chain_api                  # image: chain backend + local Chroma
├── Dockerfile.langgraph_cloud            # image: LangGraph backend + Qdrant Cloud
├── docker-compose.yml                    # one API + both Streamlit apps
├── .dockerignore                         # keeps stores/logs/venv out of the image
├── testing_rag_api.ipynb                 # scratch notebook for exercising the API
├── download_data_script/                 # fetch + parse CMS SynPUF and Synthea data
├── immobilien_dataset/                   # the five Berlin real-estate CSVs
├── chroma_immobilien/                    # persisted real-estate vector store
├── chroma_medicine/                      # persisted medical vector store
├── query_results/                        # JSON records written by the save node
├── langchain_project_venv/               # local virtualenv
└── __pycache__/
```

Plus hidden files `.env` (holds `OPENAI_API_KEY`, plus `QDRANT_URL` / `QDRANT_API_KEY` for the Qdrant-Cloud backend) and `.gitignore`.

### 📥 Data & large files (not on GitHub)

The datasets and prebuilt vector stores are too large for GitHub and are excluded via
`.gitignore`. Download them and drop them into the project root, keeping the folder names above:

**[⬇️ Download dataset + vector stores (Google Drive)](https://drive.google.com/drive/folders/1zPuuBJnQHf2nuvQiFweA_2fSNbVrvi3M?usp=drive_link)**

This covers `immobilien_dataset/`, the `download_data_script/` data, and the persisted
`chroma_immobilien/` and `chroma_medicine/` stores. With the stores in place the pipeline
runs directly; otherwise rebuild them with `build_immobilien_vectordb.py` (see
`README_vectordb.md`) and the `download_data_script/` scripts.

## ⚙️ Setup

```bash
pip install langgraph langchain-core langchain-openai langchain-chroma langchain-huggingface \
            pydantic python-dotenv sentence-transformers chromadb streamlit pandas
```

Set your OpenAI key (loaded via `python-dotenv`):

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

You also need the two Chroma stores on disk (`./chroma_immobilien` and `./chroma_medicine`)
— either from the Drive link above, or rebuilt from the scripts.

## ▶️ Run it

```python
from using_langchain_langgraph import build_pipeline

graph = build_pipeline()

result = graph.invoke({"question": "cheap 2-room rental near a U-Bahn in Kreuzberg"})
print(result["answer"])
print("saved ->", result["saved_path"], "| faithfulness:", result["evaluation"]["faithfulness"])
```

`build_pipeline()` wires everything (stores, LLM, rewriter, grader, router, answer chain,
evaluator) and returns a compiled graph. Every `invoke` drops a JSON record in
`query_results/` — which is exactly what the dashboard reads.

The router handles either domain automatically:

```python
graph.invoke({"question": "What are typical COPD reimbursement amounts?"})   # → medical
graph.invoke({"question": "new build with parking in Mitte"})                # → real_estate
```

## 🐳 Run with Docker

Two Dockerfiles are provided — **one per backend** — plus a `docker-compose.yml` that also
stands up both Streamlit apps. Pick the backend that matches where your vectors live:

| Dockerfile | Backend entrypoint | Vector store | Extra env |
|------------|--------------------|--------------|-----------|
| `Dockerfile.chain_api` | `langchain_chain_rag_function_api.py` (LangChain chain) | **local Chroma** (`./chroma_immobilien`, `./chroma_medicine`) | — |
| `Dockerfile.langgraph_cloud` | `using_langchain_langgraph_cloud.py` served via `langgraph_api.py` | **Qdrant Cloud** | `QDRANT_URL`, `QDRANT_API_KEY` |

> 🔌 `using_langchain_langgraph_cloud.py` has no web server of its own, so `langgraph_api.py`
> wraps it in the **same** `/query` + `/health` API the chain backend exposes — so
> `rag_streamlit_app.py` works against either backend unchanged.

### 1️⃣ Configure `.env`

```bash
# always
OPENAI_API_KEY=sk-...
# only for the langgraph_cloud backend
QDRANT_URL=https://<your-cluster>.qdrant.io:6333
QDRANT_API_KEY=...
```

### 2️⃣ Quickest path — `docker compose` (API + both Streamlit apps)

```bash
docker compose up --build
```

| Service | URL | What it is |
|---------|-----|------------|
| `api` | http://localhost:8000/docs | the RAG API (Swagger UI) |
| `rag_explorer` | http://localhost:8501 | `rag_streamlit_app.py` — ask questions |
| `dashboard` | http://localhost:8502 | `dashboard.py` — quality monitoring |

In the Explorer sidebar, set **API base URL** to `http://api:8000` (containers talk over the
compose network, so `localhost` won't reach the `api` service).

**Switch backend:** in `docker-compose.yml`, change the `api` service's `dockerfile:` from
`Dockerfile.chain_api` to `Dockerfile.langgraph_cloud` (and make sure `QDRANT_URL` /
`QDRANT_API_KEY` are in `.env`), then re-run `docker compose up --build`.

### 3️⃣ Or run a single backend image (no compose)

```bash
# chain + local Chroma
docker build -f Dockerfile.chain_api -t rag-chain-api .
docker run --env-file .env -p 8000:8000 \
    -v "$PWD/chroma_immobilien:/app/chroma_immobilien:ro" \
    -v "$PWD/chroma_medicine:/app/chroma_medicine:ro" \
    -v "$PWD/query_results:/app/query_results" rag-chain-api

# LangGraph + Qdrant Cloud
docker build -f Dockerfile.langgraph_cloud -t rag-langgraph-cloud .
docker run --env-file .env -p 8000:8000 \
    -v "$PWD/query_results:/app/query_results" rag-langgraph-cloud
```

Then hit the API:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "typical COPD reimbursement amounts?"}'
```

> 🗂️ The Chroma stores and `query_results/` are **mounted at runtime**, not baked into the
> image (see `.dockerignore`). For a self-contained chain image, delete the two `chroma_*`
> lines from `.dockerignore` so the stores get copied in at build time.
> 🤖 The first build pre-downloads the `sentence-transformers` embedding models — slow once,
> cached thereafter.

## 🛠️ Tech stack

**Python** · **LangChain** · **LangGraph** · **Chroma** + **Qdrant Cloud** (vector DBs) ·
**sentence-transformers** (local embeddings) · **OpenAI** `gpt-4o-mini` · **Streamlit** +
**pandas** (monitoring) · **FastAPI** (serving) · **Docker** + **docker-compose**

## 🧭 Notes & next steps

- 🔁 **Grader pass-rate:** the `save` node stores post-grade documents, so a true *docs-in
  vs docs-kept* ratio would need the pre-grade count added to each record.
- 🧷 **Match your embedding model:** the store must be built and queried with the *same*
  embedder — keep the build and query sides in sync.
- 🛰️ **Deeper tracing:** layer in Langfuse or Arize Phoenix for per-node latency and token/cost.
- 🖼️ **Inspect the graph:** `print(graph.get_graph().draw_ascii())` renders the node/edge layout.

---

<p align="center"><sub>Built while learning applied RAG, agentic workflows, and LLM evaluation. Contributions and ideas welcome. 🤝</sub></p>
