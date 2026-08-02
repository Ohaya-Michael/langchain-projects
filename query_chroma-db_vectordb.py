"""
query_chroma-db_vectordb.py
-----------------
Query the Chroma DB built by build_vectordb.py.

    python query_chroma-db_vectordb.py "cheap 2-room rental near a U-Bahn in Kreuzberg"
    python query_chroma-db_vectordb.py "family flat" --k 5 --source rentals
"""
import argparse
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION = "immobilien"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--source", help="filter by file, e.g. rentals / secondary_sales")
    args = ap.parse_args()

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # metadata filter is optional; combine with $and for more conditions
    flt = {"source": args.source} if args.source else None

    results = vectordb.similarity_search_with_score(args.query, k=args.k, filter=flt)
    for doc, score in results:
        print(f"\n[score {score:.3f}] {doc.metadata.get('id', doc.metadata.get('source'))}")
        print(doc.page_content)


if __name__ == "__main__":
    main()
