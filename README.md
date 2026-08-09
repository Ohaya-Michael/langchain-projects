# A two-domain RAG pipeline with LangGraph

Answers questions over two unrelated Chroma vector databases — Berlin real-estate
listings and medical claims — by routing each question to the right store, retrieving
and grading documents, synthesizing a sourced answer, and self-scoring it with an
LLM-as-judge. Built as a `langgraph` `StateGraph` in `using_langchain_langgraph.py`.

This is the query side of the project: it consumes the `immobilien` collection built by
`build_immobilien_vectordb.py` (see `README_vectordb.md`) plus a `medicine_claims`
collection built from the scripts in `download_data_script/`.

## Project layout

The repository is organized as follows (`tree -L 1`):

```
langchain_project/
├── README_vectordb.md                    # how the immobilien vector DB is built
├── requirements.txt                      # Python dependencies
├── __init__.py
├── build_immobilien_vectordb.py          # Berlin CSVs  -> chroma_immobilien
├── query_chroma_db_vectordb.py           # CLI to query a Chroma store directly
├── using_langchain_langgraph.py          # the LangGraph RAG pipeline (this README)
├── using_langchain_langgraph.ipynb       # notebook version of the pipeline
├── langchain_chain_rag_function_api.py   # pipeline wrapped as a callable API
├── testing_rag_api.ipynb                 # scratch notebook for exercising the API
├── download_data_script/                 # fetch + parse CMS SynPUF and Synthea data
├── immobilien_dataset/                   # the five Berlin real-estate CSVs
├── chroma_immobilien/                    # persisted real-estate vector store
├── chroma_medicine/                      # persisted medical vector store
├── query_results/                        # JSON records written by the save node
├── langchain_project_venv/               # local virtualenv
└── __pycache__/
```

Plus hidden files `.env` (holds `OPENAI_API_KEY`) and `.gitignore`. The layout above is
the expected structure — keep these names and folders in place so the relative paths in
the pipeline (`./chroma_immobilien`, `./chroma_medicine`, `./query_results`) resolve.

### Data & large files (not on GitHub)

The datasets and prebuilt vector stores are too large for GitHub and are excluded via
`.gitignore`. Download them here and drop them into the project root, keeping the folder
names above:

**[Download dataset + vector stores (Google Drive)](https://drive.google.com/drive/folders/1zPuuBJnQHf2nuvQiFweA_2fSNbVrvi3M?usp=drive_link)**

This covers `immobilien_dataset/`, the `download_data_script/` data, and the persisted
`chroma_immobilien/` and `chroma_medicine/` stores. With the stores in place you can run
the pipeline directly; otherwise rebuild them with `build_immobilien_vectordb.py` and the
`download_data_script/` scripts.

## The stores

| Store | Collection | Persist dir | What it holds |
|-------|-----------|-------------|---------------|
| `real_estate` | `immobilien` | `./chroma_immobilien` | Berlin resale, rental, new-build listings, monthly Kiez stats, transit stations |
| `medical` | `medicine_claims` | `./chroma_medicine` | `synthea` patient records + `cms_synpuf` 2008–2010 Medicare claims |

Both are embedded locally with `sentence-transformers/all-MiniLM-L6-v2` (free, no API
key). Each store is exposed as a retriever with `k=5`. The `data_source` metadata field
(`real_estate`, `synthea`, `cms_synpuf`) is what lets the answer cite where each fact
came from.

## How it works

The pipeline is a `StateGraph` over a shared `RAGState`. Each node returns a partial dict
that merges into the state, and one conditional edge picks the retriever:

```
START → rewrite → route ─┬─→ retrieve_real_estate ─┐
                         └─→ retrieve_medical ──────┴─→ grade → generate → evaluate → save → END
```

1. **rewrite** — a query rewriter turns the raw question into a clean, self-contained
   search query (fixes spelling, expands abbreviations like `COPD →
   COPD (chronic obstructive pulmonary disease)`, strips chit-chat) without changing
   meaning.
2. **route** — a structured-output router (`RouteQuery`) classifies the rewritten query
   as `medical` or `real_estate`.
3. **retrieve** — the conditional edge runs only the matching retriever (`k=5`).
4. **grade** — a binary relevance grader (`GradeDocument`, `yes`/`no`) drops off-topic
   retrievals, keeping only documents that could help answer the question.
5. **generate** — an answer chain built on `SYNTHESIS_PROMPT` synthesizes a single
   coherent answer from the graded context and tags each fact with its source, e.g.
   `(Real Estate)`, `(Synthea)`, `(CMS SynPUF)`.
6. **evaluate** — an LLM-as-judge (`RagEvaluation`) scores `faithfulness`,
   `answer_relevance`, and `context_relevance` (1–5), flags `hallucinated`, and gives a
   short `rationale`.
7. **save** — the full record (question, rewritten query, datasource, answer, evaluation,
   retrieved docs) is written to `./query_results/<timestamp>_<slug>.json`.

The LLM throughout is `gpt-4o-mini` at `temperature=0`.

## Setup

```bash
pip install langgraph langchain-core langchain-openai langchain-chroma langchain-huggingface pydantic python-dotenv sentence-transformers chromadb
```

Set your OpenAI key (loaded via `python-dotenv`):

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

You also need the two Chroma stores on disk (`./chroma_immobilien` and
`./chroma_medicine`) — build the real-estate one with `build_immobilien_vectordb.py` per
`README_vectordb.md`, and the medical one from the scripts in `download_data_script/`.

## Run it

```python
from using_langchain_langgraph import build_pipeline

graph = build_pipeline()

result = graph.invoke({"question": "cheap 2-room rental near a U-Bahn in Kreuzberg"})
print(result["answer"])
print("saved ->", result["saved_path"], "| faithfulness:", result["evaluation"]["faithfulness"])
```

`build_pipeline()` wires everything (stores, LLM, rewriter, grader, router, answer chain,
evaluator) and returns a compiled graph. Every `invoke` also drops a JSON record in
`./query_results`.

Example questions the router handles from either domain:

```python
graph.invoke({"question": "What are typical COPD reimbursement amounts?"})   # → medical
graph.invoke({"question": "new build with parking in Mitte"})                # → real_estate
```

## Notes & next steps

- **Result state:** each run returns a `RAGState` dict with `question`, `search_query`,
  `datasource`, `documents`, `context`, `answer`, `evaluation`, and `saved_path`.
- **Grading trade-off:** the grader is a coarse filter that leans toward `yes` for
  topically related docs; if it drops everything, `generate` will say it lacks enough
  context rather than guess.
- **Cross-domain safety:** the synthesis prompt is told to use only the relevant domain
  and never blend real-estate and medical content into one answer.
- **Serving it:** wrap `build_pipeline()` in a long-lived object and call `graph.invoke`
  per request to expose the pipeline behind an API (build the graph once, reuse it).
- **Inspect the graph:** `print(graph.get_graph().draw_ascii())` renders the node/edge
  layout shown above.
