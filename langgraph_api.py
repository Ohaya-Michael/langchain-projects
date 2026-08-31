"""FastAPI wrapper around the LangGraph + Qdrant-Cloud pipeline.

`using_langchain_langgraph_cloud.py` exposes `build_pipeline()` (a compiled
LangGraph graph) but no web server. This wrapper serves that graph behind the
SAME `/query` + `/health` contract that `langchain_chain_rag_function_api.py`
already serves, so `rag_streamlit_app.py` can point at either backend unchanged.

Note: the LangGraph graph always auto-routes, retrieves k=5, applies no metadata
filter, and always logs a record. The `datasource`, `filters`, `k`, and `save`
fields are therefore accepted (for request-shape compatibility with the chain
backend) but ignored here.

Run:
    uvicorn langgraph_api:app --host 0.0.0.0 --port 8000
"""

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from using_langchain_langgraph_cloud import build_pipeline


class RagEvaluation(BaseModel):
    faithfulness: int
    answer_relevance: int
    context_relevance: int
    hallucinated: bool
    rationale: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    # Accepted for API compatibility with the chain backend, but ignored:
    datasource: Optional[Literal["real_estate", "medical"]] = None
    filters: Optional[Dict[str, Any]] = None
    k: int = 5
    save: bool = True


class QueryResponse(BaseModel):
    timestamp: str
    question: str
    search_query: str
    datasource: str
    routing: str
    filters: Optional[Dict[str, Any]] = None
    answer: str
    evaluation: RagEvaluation
    retrieved: List[Dict[str, Any]]
    saved_path: Optional[str] = None


@lru_cache(maxsize=1)
def get_graph():
    """Build the LangGraph pipeline once (connects to Qdrant, loads models)."""
    return build_pipeline()


app = FastAPI(title="Two-Domain RAG API (LangGraph + Qdrant Cloud)")


@app.on_event("startup")
def _warm_up():
    get_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    final = get_graph().invoke({"question": req.question})

    docs = final.get("documents", []) or []
    return QueryResponse(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        question=final.get("question", req.question),
        search_query=final.get("search_query", ""),
        datasource=final.get("datasource", "unknown"),
        routing="auto",  # the graph always routes automatically
        filters=None,
        answer=final.get("answer", ""),
        evaluation=RagEvaluation(**(final.get("evaluation") or {})),
        retrieved=[{"content": d.page_content, "metadata": d.metadata} for d in docs],
        saved_path=final.get("saved_path"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("langgraph_api:app", host="0.0.0.0", port=8000, reload=True)
