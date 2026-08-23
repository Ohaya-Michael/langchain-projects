"""Two-domain RAG pipeline exposed as a FastAPI service.

A question is rewritten for retrieval, routed to the real-estate or medical
vector store, answered from the retrieved context with source attribution,
scored by an LLM-as-judge, and (optionally) logged to disk.

Run with:
    uvicorn rag_api:app --reload
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Dict, Any
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI, HTTPException


RESULTS_DIR = Path("./query_results")


SYNTHESIS_PROMPT = """You are a document research assistant working across two \
unrelated domains: real-estate records and medical claims. Answer the question \
using ONLY the retrieved documents below. Do not use outside knowledge or \
invent facts not present in the context.

The documents are retrieved from three sources. First decide which domain the \
question is about, then reason about the relevant source(s) accordingly:

Real-estate domain:
- "real_estate": property leases and catalog/listing documents. Good for lease \
  terms, property attributes, and listing prices. This is the only source \
  relevant to real-estate questions.

Medical domain:
- "synthea": synthetic patient records with narrative clinical notes and \
  Synthea's own internally-generated billing amounts. Good for clinical detail \
  (symptoms, history, care plans), but its dollar amounts are a simulation, \
  not real-world pricing.
- "cms_synpuf": real (de-identified/synthetic-shuffled) 2008-2010 Medicare \
  claims with ICD-9/HCPCS/DRG codes and actual historical reimbursement \
  amounts. Good for coding and real payment data, but has no narrative detail \
  and reflects 2008-2010 pricing, not current costs.

Instructions:
1. The retriever returns documents from both domains regardless of the \
   question. Use only the source(s) relevant to what's being asked and ignore \
   documents from the other domain — do not blend real-estate and medical \
   content into one answer.
2. Synthesize a single coherent answer from the relevant documents — don't \
   just list them back.
3. When you state a fact, note which source(s) it came from, e.g. \
   "(Real Estate)", "(Synthea)", or "(CMS SynPUF)".
4. If two sources within the same domain disagree (e.g. Synthea vs. CMS SynPUF \
   on cost), say so explicitly rather than picking one silently — explain that \
   the discrepancy is expected given their different pricing models and time \
   periods.
5. If the retrieved documents don't contain enough relevant information to \
   answer, say so plainly instead of guessing.
6. Keep the answer focused and avoid repeating raw metadata (IDs, dates) \
   unless they're relevant to the question.

Question: {question}

Retrieved documents:
{context}

Answer:"""


# ---------------------------------------------------------------------------
# Structured schemas
# ---------------------------------------------------------------------------
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["medical", "real_estate"] = Field(
        ...,
        description=(
            "Choose the datasource most relevant to the user's question:\n"
            "- 'medical': patient and Medicare claims data. Use for anything "
            "clinical or healthcare-billing related, including: clinical "
            "narratives, symptoms, chief complaints, care plans, and "
            "encounter/visit history (from synthetic patient records), as well "
            "as billing codes (ICD-9, HCPCS, DRG), claim payment amounts, and "
            "real-world Medicare reimbursement (from CMS SynPUF claims).\n"
            "- 'real_estate': use for questions about real estate, property "
            "listings, leases, housing prices, or related topics."
        ),
    )


class RagEvaluation(BaseModel):
    faithfulness: int = Field(..., ge=1, le=5,
        description="Is every claim grounded in the retrieved context? 5=fully supported, 1=mostly unsupported.")
    answer_relevance: int = Field(..., ge=1, le=5,
        description="How directly the answer addresses the question.")
    context_relevance: int = Field(..., ge=1, le=5,
        description="How relevant the retrieved documents are to the question.")
    hallucinated: bool = Field(...,
        description="True if the answer states facts not present in the context.")
    rationale: str = Field(..., description="One or two sentences justifying the scores.")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[source={d.metadata.get('data_source', 'unknown')}] {d.page_content}"
        for d in docs
    )


# ---------------------------------------------------------------------------
# Pipeline: built once, reused across requests
# ---------------------------------------------------------------------------
class RagPipeline:
    def __init__(self):
        load_dotenv()

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        real_estate = Chroma(
            collection_name="immobilien",
            embedding_function=embeddings,
            persist_directory="./chroma_immobilien",
        )
        medical = Chroma(
            collection_name="medicine_claims",
            embedding_function=embeddings,
            persist_directory="./chroma_medicine",
        )

        # keep the stores so we can build filtered retrievers per request
        self.stores = {"real_estate": real_estate, "medical": medical}
        self.real_estate_retriever = real_estate.as_retriever(search_kwargs={"k": 5})
        self.medical_retriever = medical.as_retriever(search_kwargs={"k": 5})

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

        # router
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at routing a user question to the "
                       "appropriate datasource. Use the description of each "
                       "datasource to decide which one is most relevant to "
                       "answer the question."),
            ("human", "{question}"),
        ])
        self.question_router = route_prompt | llm.with_structured_output(RouteQuery)

        # query rewriter
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You rewrite a user's question into a clear, "
                       "self-contained search query optimized for retrieving "
                       "documents from a vector database. Fix spelling, expand "
                       "abbreviations to their full form while keeping the "
                       "acronym (e.g. \"COPD\" -> \"COPD (chronic obstructive "
                       "pulmonary disease)\"), and strip chit-chat. Do NOT "
                       "change the meaning or add facts. Return only the "
                       "rewritten query, nothing else."),
            ("human", "{question}"),
        ])
        self.query_rewriter = rewrite_prompt | llm | StrOutputParser()

        # answer chain
        synthesis_prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
        self.answer_chain = synthesis_prompt | llm | StrOutputParser()

        # evaluator
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict evaluator of a retrieval-augmented "
                       "answer. Judge ONLY using the question, the retrieved "
                       "context, and the answer given. An answer is faithful "
                       "only if the context supports it — penalize any claim "
                       "not present in the context. Be critical and consistent."),
            ("human", "Question:\n{question}\n\nRetrieved context:\n{context}\n\nAnswer:\n{answer}"),
        ])
        self.evaluator = eval_prompt | llm.with_structured_output(RagEvaluation)

    def answer_evaluate_and_save(self, query: str, save: bool = True,
                                 out_dir=RESULTS_DIR,
                                 datasource: Optional[str] = None,
                                 filters: Optional[Dict[str, Any]] = None,
                                 k: int = 5) -> dict:
        # 1. rewrite the query for retrieval
        search_query = self.query_rewriter.invoke({"question": query})

        # 2. route (unless the caller forced a domain), then retrieve
        forced = datasource in ("real_estate", "medical")
        if forced:
            ds = datasource
        else:
            ds = self.question_router.invoke({"question": search_query}).datasource

        # build a retriever, applying a metadata filter if one was supplied
        search_kwargs: Dict[str, Any] = {"k": k}
        if filters:
            search_kwargs["filter"] = filters
        retriever = self.stores[ds].as_retriever(search_kwargs=search_kwargs)
        docs = retriever.invoke(search_query)
        context = format_docs(docs)
        datasource = ds

        # 3. answer the ORIGINAL question against the retrieved context
        answer = self.answer_chain.invoke({"question": query, "context": context})

        # 4. evaluate
        evaluation = self.evaluator.invoke(
            {"question": query, "context": context, "answer": answer}
        )
        if not isinstance(evaluation, RagEvaluation):
            evaluation = RagEvaluation.model_validate_json(evaluation.content)
        eval_dict = evaluation.model_dump()

        # 5. assemble record (and optionally log the rewritten query too)
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": query,
            "search_query": search_query,
            "datasource": datasource,
            "routing": "forced" if forced else "auto",
            "filters": filters or None,
            "answer": answer,
            "evaluation": eval_dict,
            "retrieved": [
                {"content": d.page_content, "metadata": d.metadata} for d in docs
            ],
        }

        if save:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = "".join(c if c.isalnum() else "_" for c in query).strip("_")[:40]
            path = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            record["saved_path"] = str(path)

        return record


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    """Build the pipeline once and reuse it across requests."""
    return RagPipeline()


# -----
# API
# -----
class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    datasource: Optional[Literal["real_estate", "medical"]] = Field(
        None,
        description="Force a domain. Omit / null to let the router decide automatically.",
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Chroma metadata where-clause, e.g. {\"rooms\": {\"$gte\": 3}} "
                    "or {\"data_source\": \"cms_synpuf\"}.",
    )
    k: int = Field(5, ge=1, le=25, description="Number of documents to retrieve.")
    save: bool = Field(True, description="Whether to log the result to disk.")


class QueryResponse(BaseModel):
    timestamp: str
    question: str
    search_query: str
    datasource: str
    routing: str
    filters: Optional[Dict[str, Any]] = None
    answer: str
    evaluation: RagEvaluation
    retrieved: list
    saved_path: Optional[str] = None


app = FastAPI(title="Two-Domain RAG API")


@app.on_event("startup")
def _warm_up():
    # Build models at startup so the first request isn't slow.
    get_pipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    return get_pipeline().answer_evaluate_and_save(
        req.question,
        save=req.save,
        datasource=req.datasource,
        filters=req.filters,
        k=req.k,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api:app", host="0.0.0.0", port=8000, reload=True)
