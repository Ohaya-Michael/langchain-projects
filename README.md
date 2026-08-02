# Building a vector DB from the Immobilien CSVs

Turns the five Berlin real-estate CSVs into a single Chroma vector database you can
retrieve over in a RAG pipeline.

## The data

| File | ~Rows | What it is | Becomes |
|------|------|------------|---------|
| `secondary_sales.csv` | 50,000 | Resale apartment listings | one doc per listing |
| `rentals.csv` | 30,000 | Rental listings | one doc per listing |
| `new_construction.csv` | 10,000 | New-build listings | one doc per listing |
| `kiez_prices_monthly.csv` | 6,232 | Monthly price/rent stats per Ortsteil | one doc per row |
| `transit_stations.csv` | 135 | U/S-Bahn station reference | one doc per station |

They relate through shared keys — `ortsteil` / `bezirk` tie listings to the monthly
stats, and `transit_station` / `transit_line` tie listings to the station table. Rather
than SQL-joining them, we put **all five into one collection** and tag each document with
a `source` field, so the retriever can pull the right mix (a listing plus its Kiez's
market stats, say) and you can filter by `source` when you want just one type.

## How it works

Each CSV row is converted into:

1. **`page_content`** — a natural-language sentence (this is what gets embedded and
   matched). Example:
   > *Rental apartment in Adlershof, Treptow-Köpenick, Berlin. 2-room, 55.1 m2 ...
   > Kaltmiete EUR 517/month ...*
2. **`metadata`** — the structured fields (`ortsteil`, `bezirk`, `rooms`, `price_eur`,
   `date_listed`, ...) kept as native types so you can do filtered / numeric retrieval.

Embeddings are computed locally with `sentence-transformers/all-MiniLM-L6-v2` (free, no
API key). The result is persisted to `./chroma_db`.

## Setup

```bash
pip install pandas langchain-core langchain-chroma langchain-huggingface sentence-transformers chromadb
```

## Build the DB

```bash
# quick test first — 500 rows per file
python build_vectordb.py --sample 500

# full build (~96k docs; several minutes on CPU)
python build_vectordb.py

# or just some files
python build_vectordb.py --files rentals kiez_prices_monthly
```

> First run downloads the ~90 MB embedding model. Full ingest embeds ~96k short docs;
> on a CPU expect a few minutes. Re-running appends, so delete `chroma_db/` to rebuild
> clean.

## Query it

```bash
python query_vectordb.py "cheap 2-room rental near a U-Bahn in Kreuzberg"
python query_vectordb.py "new build with parking in Mitte" --k 5 --source new_construction
```

## Use it in a RAG chain

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(collection_name="immobilien", embedding_function=emb,
            persist_directory="chroma_db")

retriever = db.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke("cheap 2-room rental near a U-Bahn in Kreuzberg")
for doc in docs:
  print(doc)
# plug `retriever` into your LangChain RetrievalQA / LCEL chain with an LLM
```

## Notes & next steps

- **Filtered retrieval:** pass a `filter=` dict, e.g. `{"source": "rentals"}` or
  `{"rooms": {"$gte": 3}}`, to scope results.
- **Scale:** 96k rows is fine for Chroma locally. If ingest is slow, start with
  `--sample` or a subset of files.
- **Better recall on numbers:** embeddings match meaning, not exact figures. For
  hard numeric constraints ("under EUR 500k"), combine vector search with the metadata
  filters rather than relying on the text alone.
