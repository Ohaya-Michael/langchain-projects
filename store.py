"""
stores.py
---------
Read-side of the RAG pipeline: open the Qdrant collections that the ingestion
scripts populated and expose them as LangChain retrievers, one per domain.

Ingestion (write side, run once):
    build_immobilien_vectordb_cloud.py  --backend qdrant   -> collection "immobilien"
    parse_cms_synpuf_for_rag_qdrant.py                      -> collection "medicine_claims"

This module NEVER writes. It only connects and retrieves.

CRITICAL: each collection must be queried with the SAME embedding model it was
ingested with. Both are 384-dim, so Qdrant won't error on a mismatch -- it will
just return poor results. The DOMAINS table below binds each collection to its
correct model.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# collection -> the embedding model it was ingested with
DOMAINS = {
    "real_estate": {
        "collection": "immobilien",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "medical": {
        "collection": "medicine_claims",
        "embed_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
}


@lru_cache(maxsize=None)
def _embeddings(model_name: str):
    """Load an embedding model once and reuse it (models are heavy)."""
    return HuggingFaceEmbeddings(model_name=model_name)


@lru_cache(maxsize=1)
def _client():
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )


def get_vectorstore(domain: str) -> QdrantVectorStore:
    """Return a read-only QdrantVectorStore for one domain."""
    cfg = DOMAINS[domain]
    return QdrantVectorStore(
        client=_client(),
        collection_name=cfg["collection"],
        embedding=_embeddings(cfg["embed_model"]),
    )


def get_retriever(domain: str, k: int = 4, filters: dict | None = None):
    """Return a retriever for one domain. `filters` maps metadata field -> value
    and is translated to a Qdrant payload filter (e.g. {"bezirk": "Mitte"})."""
    search_kwargs = {"k": k}
    if filters:
        from qdrant_client import models
        search_kwargs["filter"] = models.Filter(
            must=[models.FieldCondition(key=f"metadata.{key}",
                                        match=models.MatchValue(value=val))
                  for key, val in filters.items()]
        )
    return get_vectorstore(domain).as_retriever(search_kwargs=search_kwargs)


def build_retrievers(k: int = 4) -> dict:
    """Convenience: {domain -> retriever} for wiring into the router/graph."""
    return {domain: get_retriever(domain, k=k) for domain in DOMAINS}


if __name__ == "__main__":
    # quick smoke test: python stores.py
    retrievers = build_retrievers()
    for domain, retriever in retrievers.items():
        hits = retriever.invoke("test query")
        print(f"{domain}: {len(hits)} docs")
        if hits:
            print("  sample:", hits[0].page_content[:120], "...")