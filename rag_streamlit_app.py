"""Streamlit front-end for the two-domain RAG API (rag_api.py).

Two tabs:
  💬 Ask a question   — type a question and see the answer + evaluation.
  🎯 Domain & Filters — force a domain (real-estate / medical-claims) and
                        build metadata filters that scope retrieval.

The domain + filters chosen in the second tab are applied to the question
asked in the first tab. The app talks to the API over HTTP, so start the
backend first:

    uvicorn rag_api:app --reload      # serves http://localhost:8000
    streamlit run rag_app.py
"""

import json

import requests
import streamlit as st

# UI label  ->  API datasource value
DOMAIN_OPTIONS = {
    "🤖 Auto — let the router decide": None,
    "🏙️ Real-estate": "real_estate",
    "🩺 Medical-claims": "medical",
}
NUM_OPS = {"=": "$eq", "≥": "$gte", "≤": "$lte", ">": "$gt", "<": "$lt"}


# ---------------------------------------------------------------------------
# Filter builders (produce a Chroma metadata where-clause)
# ---------------------------------------------------------------------------
def _combine(conditions: list):
    """Combine condition dicts into a single Chroma where-clause."""
    conditions = [c for c in conditions if c]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def real_estate_filters() -> dict:
    conds = []
    source = st.selectbox(
        "Listing source",
        ["(any)", "rentals", "secondary_sales", "new_construction",
         "kiez_prices_monthly", "transit_stations"],
    )
    if source != "(any)":
        conds.append({"source": source})

    c1, c2 = st.columns(2)
    bezirk = c1.text_input("Bezirk (district)", placeholder="e.g. Mitte")
    if bezirk.strip():
        conds.append({"bezirk": bezirk.strip()})
    ortsteil = c2.text_input("Ortsteil (neighborhood)", placeholder="e.g. Kreuzberg")
    if ortsteil.strip():
        conds.append({"ortsteil": ortsteil.strip()})

    st.markdown("**Rooms**")
    r1, r2 = st.columns([1, 2])
    rooms_op = r1.selectbox("Rooms op", list(NUM_OPS), key="re_rooms_op", label_visibility="collapsed")
    rooms_on = r2.checkbox("filter by rooms")
    rooms_val = r2.number_input("Rooms value", min_value=0.0, step=1.0, value=2.0,
                                key="re_rooms_val", label_visibility="collapsed")
    if rooms_on:
        conds.append({"rooms": {NUM_OPS[rooms_op]: rooms_val}})

    st.markdown("**Price (EUR)**")
    p1, p2 = st.columns([1, 2])
    price_op = p1.selectbox("Price op", list(NUM_OPS), index=2, key="re_price_op", label_visibility="collapsed")
    price_on = p2.checkbox("filter by price")
    price_val = p2.number_input("Price value", min_value=0.0, step=10000.0, value=500000.0,
                                key="re_price_val", label_visibility="collapsed")
    if price_on:
        conds.append({"price_eur": {NUM_OPS[price_op]: price_val}})

    return _combine(conds)


def medical_filters() -> dict:
    conds = []
    c1, c2 = st.columns(2)
    data_source = c1.selectbox("Data source", ["(any)", "cms_synpuf", "synthea"])
    if data_source != "(any)":
        conds.append({"data_source": data_source})
    claim_type = c2.selectbox("Claim type", ["(any)", "inpatient", "outpatient", "carrier", "pde"])
    if claim_type != "(any)":
        conds.append({"claim_type": claim_type})

    c3, c4 = st.columns(2)
    state = c3.text_input("State code", placeholder="e.g. 33")
    if state.strip():
        conds.append({"state_code": state.strip()})
    module = c4.text_input("Disease module tag (Synthea)", placeholder="e.g. copd")
    if module.strip():
        conds.append({"disease_module_tag": module.strip()})

    st.markdown("**Claim amount ($)**")
    a1, a2 = st.columns([1, 2])
    amt_op = a1.selectbox("Amount op", list(NUM_OPS), index=1, key="med_amt_op", label_visibility="collapsed")
    amt_on = a2.checkbox("filter by amount")
    amt_val = a2.number_input("Amount value", min_value=0.0, step=1000.0, value=5000.0,
                              key="med_amt_val", label_visibility="collapsed")
    if amt_on:
        conds.append({"amount": {NUM_OPS[amt_op]: amt_val}})

    return _combine(conds)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_api(base_url: str, payload: dict, timeout: int = 120) -> dict:
    url = base_url.rstrip("/") + "/query"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def render_result(res: dict):
    top = st.columns(3)
    top[0].metric("Domain used", res.get("datasource", "?"))
    top[1].metric("Routing", res.get("routing", "?"))
    ev = res.get("evaluation", {}) or {}
    top[2].metric("Faithfulness", ev.get("faithfulness", "?"))

    st.markdown("### Answer")
    st.write(res.get("answer", ""))

    m = st.columns(4)
    m[0].metric("Answer rel.", ev.get("answer_relevance", "?"))
    m[1].metric("Context rel.", ev.get("context_relevance", "?"))
    m[2].metric("Hallucinated", "⚠️ yes" if ev.get("hallucinated") else "no")
    m[3].metric("Docs retrieved", len(res.get("retrieved", []) or []))
    if ev.get("rationale"):
        st.caption("Judge rationale: " + ev["rationale"])

    if res.get("filters"):
        st.markdown("**Filters applied:**")
        st.code(json.dumps(res["filters"], indent=2), language="json")

    retrieved = res.get("retrieved", []) or []
    with st.expander(f"Retrieved documents ({len(retrieved)})"):
        if not retrieved:
            st.info("No documents matched — try loosening the filters or switching domain.")
        for i, d in enumerate(retrieved, 1):
            src = (d.get("metadata") or {}).get("data_source", "unknown")
            st.markdown(f"**{i}. [{src}]**")
            st.write(d.get("content", ""))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="RAG Explorer", page_icon="🔍", layout="wide")
    st.title("🔍 Two-Domain RAG Explorer")
    st.caption("Ask across Berlin real-estate listings and medical claims — with optional domain and filters.")

    # session defaults
    st.session_state.setdefault("datasource", None)
    st.session_state.setdefault("filters", None)
    st.session_state.setdefault("domain_label", list(DOMAIN_OPTIONS)[0])

    # --- sidebar: connection ---
    st.sidebar.header("⚙️ API connection")
    base_url = st.sidebar.text_input("API base URL", value="http://localhost:8000")
    k = st.sidebar.slider("Documents to retrieve (k)", 1, 25, 5)
    save = st.sidebar.checkbox("Log result on the server", value=True)
    if st.sidebar.button("Check API health"):
        try:
            h = requests.get(base_url.rstrip("/") + "/health", timeout=10)
            st.sidebar.success(f"OK · {h.json()}") if h.ok else st.sidebar.error(f"HTTP {h.status_code}")
        except requests.RequestException as e:
            st.sidebar.error(f"Not reachable: {e}")

    tab_ask, tab_domain = st.tabs(["💬 Ask a question", "🎯 Domain & Filters"])

    # --- Domain & Filters tab (rendered first so the Ask tab sees the values) ---
    with tab_domain:
        st.subheader("Choose a domain")
        label = st.radio(
            "Which knowledge base should answer?",
            list(DOMAIN_OPTIONS),
            index=list(DOMAIN_OPTIONS).index(st.session_state["domain_label"]),
        )
        st.session_state["domain_label"] = label
        st.session_state["datasource"] = DOMAIN_OPTIONS[label]

        st.divider()
        st.subheader("Metadata filters")
        st.caption("Optional. Scope retrieval to documents whose metadata matches.")

        if st.session_state["datasource"] == "real_estate":
            built = real_estate_filters()
        elif st.session_state["datasource"] == "medical":
            built = medical_filters()
        else:
            built = None
            st.info("Filters are domain-specific. Pick **Real-estate** or **Medical-claims** above to build filters, "
                    "or use the advanced raw filter below.")

        with st.expander("Advanced — raw JSON filter (overrides the builder)"):
            raw = st.text_area(
                "Chroma where-clause",
                placeholder='{"rooms": {"$gte": 3}}   or   {"data_source": "cms_synpuf"}',
                height=90,
            )
            if raw.strip():
                try:
                    built = json.loads(raw)
                    st.success("Using raw JSON filter.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

        st.session_state["filters"] = built

        st.divider()
        st.markdown("**Current selection**")
        st.write(f"Domain: `{st.session_state['datasource'] or 'auto'}`")
        st.code(json.dumps(st.session_state["filters"], indent=2) if st.session_state["filters"]
                else "(no filters)", language="json")

    # --- Ask tab ---
    with tab_ask:
        ds = st.session_state["datasource"]
        filt = st.session_state["filters"]
        chips = f"Domain: **{ds or 'auto'}**"
        chips += f" · Filters: **{'yes' if filt else 'none'}**"
        st.markdown(chips)

        question = st.text_area(
            "Your question",
            placeholder="e.g. cheap 2-room rental near a U-Bahn in Kreuzberg",
            height=100,
        )
        ask = st.button("Ask", type="primary")

        if ask:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                payload = {
                    "question": question.strip(),
                    "datasource": ds,
                    "filters": filt,
                    "k": k,
                    "save": save,
                }
                try:
                    with st.spinner("Querying the RAG pipeline…"):
                        res = call_api(base_url, payload)
                    render_result(res)
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Could not reach the API at {base_url}. "
                        "Start it with `uvicorn rag_api:app --reload` and check the URL in the sidebar."
                    )
                except requests.HTTPError as e:
                    st.error(f"API returned an error: {e} — {getattr(e.response, 'text', '')}")
                except requests.RequestException as e:
                    st.error(f"Request failed: {e}")


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _has_streamlit_context() or __name__ == "__main__":
    main()
