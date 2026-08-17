"""Streamlit dashboard over the RAG pipeline's ./query_results/*.json logs.

Each log is one record written by the pipeline's `save` node:
    {timestamp, question, search_query, datasource, answer,
     evaluation: {faithfulness, answer_relevance, context_relevance,
                  hallucinated, rationale},
     retrieved: [{content, metadata}, ...]}

Run:
    pip install streamlit pandas
    streamlit run dashboard.py
    # or point at a different folder:
    streamlit run dashboard.py -- --log-dir ./query_results
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_LOG_DIR = "./query_results"
METRICS = ["faithfulness", "answer_relevance", "context_relevance"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
    "context_relevance": "Context relevance",
}


# ---------------------------------------------------------------------------
# Data loading + aggregation (pure functions, no Streamlit — easy to test)
# ---------------------------------------------------------------------------
def load_records(log_dir) -> list:
    """Read every *.json in log_dir into a list of dicts, skipping bad files."""
    records = []
    for p in sorted(Path(log_dir).glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            if isinstance(rec, dict):
                rec["_file"] = p.name
                records.append(rec)
        except (json.JSONDecodeError, OSError):
            continue
    return records


def records_to_df(records: list) -> pd.DataFrame:
    """Flatten records into one row per query with the evaluation fields lifted out."""
    rows = []
    for r in records:
        ev = r.get("evaluation") or {}
        retrieved = r.get("retrieved") or []
        try:
            ts = pd.to_datetime(r.get("timestamp"))
        except (ValueError, TypeError):
            ts = pd.NaT
        rows.append({
            "timestamp": ts,
            "question": r.get("question", ""),
            "search_query": r.get("search_query", ""),
            "datasource": r.get("datasource", "unknown"),
            "faithfulness": ev.get("faithfulness"),
            "answer_relevance": ev.get("answer_relevance"),
            "context_relevance": ev.get("context_relevance"),
            "hallucinated": ev.get("hallucinated"),
            "rationale": ev.get("rationale", ""),
            "answer": r.get("answer", ""),
            "n_retrieved": len(retrieved),
            "retrieved": retrieved,
            "file": r.get("_file", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        for m in METRICS:
            df[m] = pd.to_numeric(df[m], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """Headline metrics for the KPI row."""
    k = {"total": int(len(df))}
    for m in METRICS:
        k[f"avg_{m}"] = float(df[m].mean()) if len(df) else float("nan")
    if len(df):
        h = df["hallucinated"].dropna()
        k["hallucination_rate"] = float((h == True).mean()) if len(h) else float("nan")  # noqa: E712
        k["no_context_rate"] = float((df["n_retrieved"] == 0).mean())
    else:
        k["hallucination_rate"] = float("nan")
        k["no_context_rate"] = float("nan")
    return k


def score_distribution(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Count of each 1..5 score for a metric (missing buckets shown as 0)."""
    counts = df[metric].dropna().astype(int).value_counts()
    counts = counts.reindex(range(1, 6), fill_value=0).sort_index()
    return counts.rename("count").rename_axis("score").reset_index()


def daily_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day average of each metric, for the trend chart."""
    if df.empty or df["timestamp"].isna().all():
        return pd.DataFrame(columns=METRICS)
    d = df.dropna(subset=["timestamp"]).copy()
    d["day"] = d["timestamp"].dt.floor("D")
    return d.groupby("day")[METRICS].mean()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _cli_default_log_dir() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    args, _ = parser.parse_known_args()
    return args.log_dir


def main():
    st.set_page_config(page_title="RAG Monitoring", page_icon="📊", layout="wide")
    st.title("📊 RAG pipeline monitoring")
    st.caption("Reads the JSON records written by the pipeline's `save` node.")

    # --- sidebar controls ---
    st.sidebar.header("Data")
    log_dir = st.sidebar.text_input("Log directory", value=_cli_default_log_dir())

    records = load_records(log_dir)
    df = records_to_df(records)

    if df.empty:
        st.warning(
            f"No JSON logs found in `{log_dir}`. Run some queries through the "
            "pipeline first, or point the sidebar at the right folder."
        )
        st.stop()

    # --- filters ---
    st.sidebar.header("Filters")
    sources = sorted(df["datasource"].dropna().unique().tolist())
    picked = st.sidebar.multiselect("Datasource", sources, default=sources)
    df = df[df["datasource"].isin(picked)]

    valid_ts = df["timestamp"].dropna()
    if not valid_ts.empty:
        min_d, max_d = valid_ts.min().date(), valid_ts.max().date()
        if min_d < max_d:
            start, end = st.sidebar.slider(
                "Date range", min_value=min_d, max_value=max_d, value=(min_d, max_d)
            )
            mask = df["timestamp"].dt.date.between(start, end) | df["timestamp"].isna()
            df = df[mask]

    only_halluc = st.sidebar.checkbox("Only hallucinated answers", value=False)
    if only_halluc:
        df = df[df["hallucinated"] == True]  # noqa: E712

    if df.empty:
        st.info("No records match the current filters.")
        st.stop()

    # --- KPI row ---
    k = compute_kpis(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Queries", k["total"])
    c2.metric("Avg faithfulness", f"{k['avg_faithfulness']:.2f}")
    c3.metric("Avg answer rel.", f"{k['avg_answer_relevance']:.2f}")
    c4.metric("Avg context rel.", f"{k['avg_context_relevance']:.2f}")
    c5.metric("Hallucination rate", f"{k['hallucination_rate']*100:.0f}%")
    c6.metric("No-context answers", f"{k['no_context_rate']*100:.0f}%")

    st.divider()

    # --- trend + datasource split ---
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Scores over time (daily average)")
        dm = daily_means(df)
        if dm.empty:
            st.write("Not enough timestamped data to plot a trend.")
        else:
            dm = dm.rename(columns=METRIC_LABELS)
            st.line_chart(dm, height=280)
    with right:
        st.subheader("Datasource split")
        split = df["datasource"].value_counts().rename_axis("datasource").reset_index(name="count")
        st.bar_chart(split, x="datasource", y="count", height=280)

    # --- score distributions ---
    st.subheader("Score distributions (1–5)")
    dcols = st.columns(3)
    for col, metric in zip(dcols, METRICS):
        with col:
            st.caption(METRIC_LABELS[metric])
            dist = score_distribution(df, metric)
            st.bar_chart(dist, x="score", y="count", height=220)

    # --- hallucinations by datasource ---
    st.subheader("Hallucinations by datasource")
    hb = (
        df.assign(hallucinated=df["hallucinated"] == True)  # noqa: E712
        .groupby("datasource")["hallucinated"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "hallucinated", "count": "total"})
    )
    hb["rate %"] = (hb["hallucinated"] / hb["total"] * 100).round(0)
    st.dataframe(hb)

    st.divider()

    # --- recent queries table ---
    st.subheader("Recent queries")
    show = df.sort_values("timestamp", ascending=False)
    table = show[[
        "timestamp", "datasource", "question",
        "faithfulness", "answer_relevance", "context_relevance",
        "hallucinated", "n_retrieved",
    ]]
    st.dataframe(table, hide_index=True)

    # --- drill-down ---
    st.subheader("Inspect a query")
    labels = [
        f"{r.timestamp}  ·  [{r.datasource}]  {str(r.question)[:70]}"
        for r in show.itertuples()
    ]
    if labels:
        idx = st.selectbox("Pick a query", range(len(labels)), format_func=lambda i: labels[i])
        row = show.iloc[idx]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Faithfulness", row["faithfulness"])
        m2.metric("Answer rel.", row["answer_relevance"])
        m3.metric("Context rel.", row["context_relevance"])
        m4.metric("Hallucinated", "yes" if row["hallucinated"] else "no")
        st.markdown(f"**Question:** {row['question']}")
        if row["search_query"]:
            st.markdown(f"**Rewritten query:** {row['search_query']}")
        st.markdown("**Answer:**")
        st.write(row["answer"])
        if row["rationale"]:
            st.markdown(f"**Judge rationale:** {row['rationale']}")
        with st.expander(f"Retrieved documents ({row['n_retrieved']})"):
            for i, doc in enumerate(row["retrieved"], 1):
                src = (doc.get("metadata") or {}).get("data_source", "unknown")
                st.markdown(f"**{i}. [{src}]**")
                st.write(doc.get("content", ""))


def _has_streamlit_context() -> bool:
    """True when executed by `streamlit run` (a script-run context exists)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# Run under `streamlit run dashboard.py` (context present) or `python dashboard.py`
# (__main__), but NOT when the module is merely imported for its helper functions.
if _has_streamlit_context() or __name__ == "__main__":
    main()