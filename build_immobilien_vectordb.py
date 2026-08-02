"""
build_immobilien_vectordb.py
-----------------
Build a Chroma vector database from the Berlin real-estate CSVs for a RAG project.

Strategy:
  * Each row of each CSV becomes ONE LangChain Document.
  * page_content is a natural-language sentence (this is what gets embedded /
    what the retriever matches against).
  * metadata holds the structured fields (ortsteil, bezirk, price, rooms, ...)
    so you can do filtered retrieval later (e.g. only rentals in Mitte).

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, free, no API key).
Vector store: Chroma, persisted to ./chroma_db.

Usage:
    python build_immobilien_vectordb.py                # ingest everything (~96k rows)
    python build_immobilien_vectordb.py --sample 500   # ingest only 500 rows per file (fast test)
    python build_immobilien_vectordb.py --files rentals secondary_sales
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ----------
# Config
# ----------
DATA_DIR = Path(__file__).parent / "immobilien_dataset"
PERSIST_DIR = "chroma_immobilien"
COLLECTION = "immobilien"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 1000  # docs embedded + written per batch


# ----------
# Helpers
# ----------
def clean_meta(d: dict) -> dict:
    """Chroma only accepts str/int/float/bool metadata values (no NaN/None)."""
    out = {}
    for k, v in d.items():
        # normalize numpy scalars to native python types
        if isinstance(v, np.generic):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="rows per file (omit for all)")
    ap.add_argument("--files", nargs="*", default=list(FILES.keys()),
                    help="subset of files to ingest")
    args = ap.parse_args()

    print("Building documents...")
    docs = build_documents(args.files, args.sample)
    print(f"Total documents: {len(docs)}")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Writing to Chroma at {PERSIST_DIR} (collection '{COLLECTION}')")
    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        vectordb.add_documents(batch)
        print(f"  embedded {min(i + BATCH_SIZE, len(docs))}/{len(docs)}")

    print("Done. Vector DB persisted.")


if __name__ == "__main__":
    main()
