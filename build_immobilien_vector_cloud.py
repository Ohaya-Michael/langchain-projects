"""
build_immobilien_vectordb_cloud.py
----------------------------------
Build a CLOUD vector database from the Berlin real-estate CSVs for a RAG project.

Same document-building logic as the original Chroma script, but the vectors are
written to a MANAGED vector store so the deployed API (Hugging Face Space, etc.)
can query them without shipping a local ./chroma_* folder.

Embeddings : sentence-transformers/all-MiniLM-L6-v2  (local, free, 384-dim)
Vector DB  : Pinecone (serverless)  OR  Qdrant Cloud   -- selectable at runtime

Env vars (put in a .env file):
    # Pinecone
    PINECONE_API_KEY=...
    PINECONE_INDEX=adaptive-rag        # optional, defaults below
    # Qdrant Cloud
    QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
    QDRANT_API_KEY=...

Install:
    pip install pandas numpy python-dotenv langchain-core langchain-huggingface \
                langchain-pinecone pinecone langchain-qdrant qdrant-client

Usage:
    # Pinecone, full ingest (~96k rows)
    python build_immobilien_vectordb_cloud.py --backend pinecone

    # Qdrant Cloud, quick 500-row test
    python build_immobilien_vectordb_cloud.py --backend qdrant --sample 500

    # subset of files
    python build_immobilien_vectordb_cloud.py --backend pinecone --files rentals secondary_sales
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ----------
# Config
# ----------
DATA_DIR = Path(__file__).parent / "immobilien_dataset"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384          # all-MiniLM-L6-v2 -> 384; MUST match the index/collection
DISTANCE = "cosine"      # MiniLM works well with cosine
NAMESPACE = "immobilien"      # Pinecone namespace  (was Chroma collection)
COLLECTION = "immobilien"     # Qdrant collection    (was Chroma collection)
DEFAULT_INDEX = "adaptive-rag"  # Pinecone index name (shared across domains)
BATCH_SIZE = 256         # docs embedded + upserted per batch (safe for Pinecone req size)


# ----------
# Helpers
# ----------
def clean_meta(d: dict) -> dict:
    """Managed stores accept str/int/float/bool metadata (no NaN/None/nested)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):          # normalize numpy scalars
            v = v.item()
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        if isinstance(v, (int, float, bool, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def yn(v) -> str:
    return "yes" if str(v).lower() in ("true", "1", "yes") else "no"


# -----------------------------------------------------
# Row -> natural-language text, one function per file
# (unchanged from the original script)
# -----------------------------------------------------
def secondary_row_to_doc(r):
    text = (
        f"For-sale apartment (resale) in {r['ortsteil']}, {r['bezirk']}, Berlin. "
        f"{r['rooms']}-room {r['property_type']}, {r['area_m2']} m2, on floor {r['floor']} "
        f"of {r['total_floors']}, built {r['year_built']} ({r['building_era']}). "
        f"Condition: {r['condition']}, position: {r['position']}, energy class {r['energy_class']}. "
        f"Lift: {yn(r['has_lift'])}, balcony: {yn(r['has_balcony'])}, "
        f"cellar: {yn(r['has_cellar'])}, parking: {yn(r['has_parking'])}. "
        f"Nearest transit: {r['transit_station']} ({r['transit_line']}), "
        f"{r['transit_distance_min']} min away; {r['to_brandenburg_gate_km']} km to Brandenburg Gate. "
        f"Price: EUR {r['price_eur']:,} ({r['price_per_m2_eur']:,} EUR/m2). "
        f"Listed {r['date_listed']} at a mortgage rate of {r['mortgage_rate_at_listing']}%."
    )
    meta = clean_meta({
        "source": "secondary_sales", "id": r["id"], "listing_type": "resale",
        "ortsteil": r["ortsteil"], "bezirk": r["bezirk"], "kiez_premium": r["kiez_premium"],
        "property_type": r["property_type"], "rooms": r["rooms"], "area_m2": r["area_m2"],
        "year_built": r["year_built"], "energy_class": r["energy_class"],
        "price_eur": r["price_eur"], "price_per_m2_eur": r["price_per_m2_eur"],
        "date_listed": r["date_listed"],
    })
    return Document(page_content=text, metadata=meta)


def rental_row_to_doc(r):
    text = (
        f"Rental apartment in {r['ortsteil']}, {r['bezirk']}, Berlin. "
        f"{r['rooms']}-room {r['property_type']}, {r['area_m2']} m2, floor {r['floor']} of {r['total_floors']}, "
        f"{r['building_era']}, condition {r['condition']}, position {r['position']}, energy class {r['energy_class']}. "
        f"Furnished: {yn(r['furnished'])}, lift: {yn(r['has_lift'])}, balcony: {yn(r['has_balcony'])}. "
        f"Nearest transit: {r['transit_station']} ({r['transit_line']}), {r['transit_distance_min']} min; "
        f"{r['to_brandenburg_gate_km']} km to Brandenburg Gate. "
        f"Kaltmiete EUR {r['kaltmiete_eur_monthly']}/month, Nebenkosten EUR {r['nebenkosten_eur_monthly']}, "
        f"Warmmiete EUR {r['warmmiete_eur_monthly']}/month ({r['rent_per_m2_kalt_eur']} EUR/m2 kalt). "
        f"Deposit: {r['kaution_months']} months. Listed {r['date_listed']}."
    )
    meta = clean_meta({
        "source": "rentals", "id": r["id"], "listing_type": "rental",
        "ortsteil": r["ortsteil"], "bezirk": r["bezirk"], "kiez_premium": r["kiez_premium"],
        "property_type": r["property_type"], "rooms": r["rooms"], "area_m2": r["area_m2"],
        "energy_class": r["energy_class"], "furnished": str(r["furnished"]),
        "kaltmiete_eur_monthly": r["kaltmiete_eur_monthly"],
        "warmmiete_eur_monthly": r["warmmiete_eur_monthly"],
        "rent_per_m2_kalt_eur": r["rent_per_m2_kalt_eur"], "date_listed": r["date_listed"],
    })
    return Document(page_content=text, metadata=meta)


def new_construction_row_to_doc(r):
    text = (
        f"New-construction apartment in {r['ortsteil']}, {r['bezirk']}, Berlin. "
        f"Project '{r['project_name']}' by developer {r['developer']} "
        f"({r['total_project_units']} units, completion {r['completion_year']}, status {r['possession_status']}). "
        f"{r['rooms']}-room {r['property_type']}, {r['area_m2']} m2, floor {r['floor']} of {r['total_floors']}, "
        f"energy class {r['energy_class']}. Lift: {yn(r['has_lift'])}, balcony: {yn(r['has_balcony'])}, "
        f"parking: {yn(r['has_parking'])}. Payment plan: {r['payment_plan']}. "
        f"Nearest transit: {r['transit_station']} ({r['transit_line']}), {r['transit_distance_min']} min; "
        f"{r['to_brandenburg_gate_km']} km to Brandenburg Gate. "
        f"Price: EUR {r['price_eur']:,} ({r['price_per_m2_eur']:,} EUR/m2). Listed {r['date_listed']}."
    )
    meta = clean_meta({
        "source": "new_construction", "id": r["id"], "listing_type": "new_construction",
        "project_name": r["project_name"], "developer": r["developer"],
        "ortsteil": r["ortsteil"], "bezirk": r["bezirk"],
        "property_type": r["property_type"], "rooms": r["rooms"], "area_m2": r["area_m2"],
        "energy_class": r["energy_class"], "completion_year": r["completion_year"],
        "price_eur": r["price_eur"], "price_per_m2_eur": r["price_per_m2_eur"],
        "date_listed": r["date_listed"],
    })
    return Document(page_content=text, metadata=meta)


def kiez_row_to_doc(r):
    text = (
        f"Monthly market statistics for {r['ortsteil']}, {r['bezirk']}, Berlin in {r['year_month']}. "
        f"Kiez premium: {r['kiez_premium']}. "
        f"Resale price {r['secondary_price_per_m2_eur']} EUR/m2, "
        f"new-construction price {r['new_construction_price_per_m2_eur']} EUR/m2, "
        f"average cold rent {r['kaltmiete_per_m2_monthly_eur']} EUR/m2/month. "
        f"Listings this month: {r['n_listings_secondary']} resale, "
        f"{r['n_listings_new_construction']} new-build, {r['n_listings_rental']} rental. "
        f"ECB main rate {r['ecb_main_rate_pct']}%, average mortgage rate {r['avg_mortgage_rate_pct']}%."
    )
    meta = clean_meta({
        "source": "kiez_prices_monthly", "listing_type": "market_stats",
        "year_month": r["year_month"], "ortsteil": r["ortsteil"], "bezirk": r["bezirk"],
        "kiez_premium": r["kiez_premium"],
        "secondary_price_per_m2_eur": r["secondary_price_per_m2_eur"],
        "kaltmiete_per_m2_monthly_eur": r["kaltmiete_per_m2_monthly_eur"],
        "avg_mortgage_rate_pct": r["avg_mortgage_rate_pct"],
    })
    return Document(page_content=text, metadata=meta)


def transit_row_to_doc(r):
    text = (
        f"Transit station '{r['station_name']}' on line {r['line']} in Berlin. "
        f"Opened {r['year_opened']}, located at ({r['lat']}, {r['lon']}), "
        f"{r['to_brandenburg_gate_km']} km from the Brandenburg Gate."
    )
    meta = clean_meta({
        "source": "transit_stations", "listing_type": "transit",
        "station_name": r["station_name"], "line": r["line"],
        "year_opened": r["year_opened"],
        "to_brandenburg_gate_km": r["to_brandenburg_gate_km"],
    })
    return Document(page_content=text, metadata=meta)


FILES = {
    "secondary_sales":     ("secondary_sales.csv",     secondary_row_to_doc),
    "rentals":             ("rentals.csv",             rental_row_to_doc),
    "new_construction":    ("new_construction.csv",    new_construction_row_to_doc),
    "kiez_prices_monthly": ("kiez_prices_monthly.csv", kiez_row_to_doc),
    "transit_stations":    ("transit_stations.csv",    transit_row_to_doc),
}


# ------------------
# Build documents
# ------------------
def build_documents(which, sample=None):
    docs = []
    for name in which:
        fname, fn = FILES[name]
        df = pd.read_csv(DATA_DIR / fname)
        if sample:
            df = df.head(sample)
        for _, row in df.iterrows():
            docs.append(fn(row))
        print(f"  {name}: {len(df)} rows -> documents")
    return docs


# -------------------------------------------------
# Backends: return a LangChain VectorStore, creating
# the index/collection if it does not exist yet.
# -------------------------------------------------
def get_pinecone_store(embeddings):
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    api_key = os.environ["PINECONE_API_KEY"]
    index_name = os.environ.get("PINECONE_INDEX", DEFAULT_INDEX)

    pc = Pinecone(api_key=api_key)
    if not pc.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}' (dim={EMBED_DIM}, {DISTANCE})")
        pc.create_index(
            name=index_name,
            dimension=EMBED_DIM,
            metric=DISTANCE,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    print(f"Using Pinecone index '{index_name}', namespace '{NAMESPACE}'")
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=NAMESPACE,
    )


def get_qdrant_store(embeddings):
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    from langchain_qdrant import QdrantVectorStore

    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]

    client = QdrantClient(url=url, api_key=api_key)
    if not client.collection_exists(COLLECTION):
        print(f"Creating Qdrant collection '{COLLECTION}' (size={EMBED_DIM}, {DISTANCE})")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    print(f"Using Qdrant collection '{COLLECTION}'")
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )


def get_store(backend, embeddings):
    if backend == "pinecone":
        return get_pinecone_store(embeddings)
    if backend == "qdrant":
        return get_qdrant_store(embeddings)
    raise ValueError(f"unknown backend: {backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["pinecone", "qdrant"], required=True,
                    help="managed vector store to write to")
    ap.add_argument("--sample", type=int, default=None,
                    help="rows per file (omit for all)")
    ap.add_argument("--files", nargs="*", default=list(FILES.keys()),
                    help="subset of files to ingest")
    args = ap.parse_args()

    load_dotenv()

    print("Building documents...")
    docs = build_documents(args.files, args.sample)
    print(f"Total documents: {len(docs)}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Connecting to backend: {args.backend}")
    store = get_store(args.backend, embeddings)

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        store.add_documents(batch)
        print(f"  embedded {min(i + BATCH_SIZE, len(docs))}/{len(docs)}")

    print("Done. Vectors written to the cloud store.")


if __name__ == "__main__":
    main()