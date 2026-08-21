### Router
import os
import json
from typing import Literal, TypedDict, List
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv


# -------------------------------------------------
# Setup: embeddings, vector stores, LLM, prompts
# -------------------------------------------------
def build_stores():
    load_dotenv()

    # Each collection MUST be queried with the model it was ingested with.
    re_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    med_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],    
        api_key=os.environ["QDRANT_API_KEY"],
    )

    real_estate = QdrantVectorStore(
        client=client,
        collection_name="immobilien",
        embedding=re_embeddings,
    )
    medical = QdrantVectorStore(
        client=client,
        collection_name="medicine_claims",
        embedding=med_embeddings,
    )

    return re_embeddings, real_estate, medical


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


def build_llm():
    # LLM with function call
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0)
    return llm


def build_query_rewriter(llm):
    # --- query rewriter: raw question -> retrieval-optimized search query ---
    rewrite_system = """You rewrite a user's question into a clear, self-contained \
search query optimized for retrieving documents from a vector database. Fix \
spelling, expand abbreviations to their full form while keeping the acronym \
(e.g. "COPD" -> "COPD (chronic obstructive pulmonary disease)"), and strip \
chit-chat. Do NOT change the meaning or add facts. Return only the rewritten \
query, nothing else."""

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", rewrite_system),
        ("human", "{question}"),
    ])

    query_rewriter = rewrite_prompt | llm | StrOutputParser()
    return query_rewriter


# -----------------------------
# Document relevance grader
# -----------------------------
# --- document relevance grader: is a retrieved doc relevant to the question? ---
class GradeDocument(BaseModel):
    """Binary relevance score for a single retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Is the document relevant to the question? 'yes' or 'no'.",
    )


def build_document_grader(llm):
    grade_system = """You are a grader assessing whether a retrieved document is \
relevant to a user's question. Grade it 'yes' if the document contains \
keywords, facts, or semantic meaning that could help answer the question; \
grade 'no' if it is about an unrelated subject or a different domain. This is a \
coarse filter to drop irrelevant retrievals, not a strict test — lean toward \
'yes' for topically related documents, but reject clear mismatches (e.g. a \
real-estate lease for a medical question). Respond with a single binary score."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", grade_system),
        ("human", "Question:\n{question}\n\nRetrieved document:\n{document}"),
    ])

    structured_grader = llm.with_structured_output(GradeDocument, method="function_calling")
    document_grader = grade_prompt | structured_grader
    return document_grader


def grade_documents(document_grader, question, docs):
    """Return only the documents graded relevant to the question."""
    graded = document_grader.batch(
        [{"question": question, "document": d.page_content} for d in docs]
    )
    return [d for d, g in zip(docs, graded) if g.binary_score == "yes"]


# ------------
# Router
# ------------
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


def build_router(llm, real_estate, medical):
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert at routing a user question to the appropriate \
datasource. Use the description of each datasource to decide which one is \
most relevant to answer the question."""

    route_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    question_router = route_prompt | structured_llm_router
    real_estate_retriever = real_estate.as_retriever(search_kwargs={"k": 5})
    medical_retriever = medical.as_retriever(search_kwargs={"k": 5})
    return question_router, real_estate_retriever, medical_retriever


# -----------------------------------------
# Answer chain + LLM-as-judge evaluator
# -----------------------------------------
RESULTS_DIR = Path("./query_results")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[source={d.metadata.get('data_source', 'unknown')}] {d.page_content}"
        for d in docs
    )


def build_answer_chain(llm):
    # --- answer chain built from your existing SYNTHESIS_PROMPT string ---
    synthesis_prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    answer_chain = synthesis_prompt | llm | StrOutputParser()
    return answer_chain


# --- LLM-as-judge evaluator ---
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


def build_evaluator(llm):
    eval_llm = llm.with_structured_output(RagEvaluation)

    EVAL_SYSTEM = """You are a strict evaluator of a retrieval-augmented answer. \
Judge ONLY using the question, the retrieved context, and the answer given. \
An answer is faithful only if the context supports it — penalize any claim not \
present in the context. Be critical and consistent."""

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", EVAL_SYSTEM),
        ("human", "Question:\n{question}\n\nRetrieved context:\n{context}\n\nAnswer:\n{answer}"),
    ])
    evaluator = eval_prompt | eval_llm
    return evaluator


# ---------------------
# Graph state + nodes
# ---------------------
# --- shared state passed between nodes ---
class RAGState(TypedDict):
    question: str        # original user question
    search_query: str    # rewritten query
    datasource: str
    documents: List[Document]
    context: str
    answer: str
    evaluation: dict
    saved_path: str


def build_graph(query_rewriter, question_router, real_estate_retriever,
                medical_retriever, document_grader, answer_chain, evaluator):
    # --- nodes: each returns a partial dict that merges into state ---
    def rewrite_node(state: RAGState) -> dict:
        return {"search_query": query_rewriter.invoke({"question": state["question"]})}

    def route_node(state: RAGState) -> dict:
        ds = question_router.invoke({"question": state["search_query"]}).datasource
        return {"datasource": ds}

    def retrieve_real_estate(state: RAGState) -> dict:
        docs = real_estate_retriever.invoke(state["search_query"])
        return {"documents": docs, "context": format_docs(docs)}

    def retrieve_medical(state: RAGState) -> dict:
        docs = medical_retriever.invoke(state["search_query"])
        return {"documents": docs, "context": format_docs(docs)}

    def grade_node(state: RAGState) -> dict:
        relevant = grade_documents(document_grader, state["question"], state["documents"])
        return {"documents": relevant, "context": format_docs(relevant)}

    def generate_node(state: RAGState) -> dict:
        answer = answer_chain.invoke({"question": state["question"], "context": state["context"]})
        return {"answer": answer}

    def evaluate_node(state: RAGState) -> dict:
        evaluation = evaluator.invoke({
            "question": state["question"],
            "context": state["context"],
            "answer": state["answer"],
        })
        if not isinstance(evaluation, RagEvaluation):
            evaluation = RagEvaluation.model_validate_json(evaluation.content)
        return {"evaluation": evaluation.model_dump()}

    def save_node(state: RAGState) -> dict:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "question": state["question"],
            "search_query": state["search_query"],
            "datasource": state["datasource"],
            "answer": state["answer"],
            "evaluation": state["evaluation"],
            "retrieved": [
                {"content": d.page_content, "metadata": d.metadata}
                for d in state["documents"]
            ],
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in state["question"]).strip("_")[:40]
        path = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return {"saved_path": str(path)}

    # --- routing is a conditional edge that reads the datasource set by route_node ---
    def pick_retriever(state: RAGState) -> str:
        return state["datasource"]

    # --- wire the graph ---
    builder = StateGraph(RAGState)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("route", route_node)
    builder.add_node("retrieve_real_estate", retrieve_real_estate)
    builder.add_node("retrieve_medical", retrieve_medical)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("save", save_node)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "route")
    builder.add_conditional_edges(
        "route",
        pick_retriever,
        {"real_estate": "retrieve_real_estate", "medical": "retrieve_medical"},
    )
    # builder.add_edge("retrieve_real_estate", "generate")
    # builder.add_edge("retrieve_medical", "generate")
    builder.add_edge("retrieve_real_estate", "grade")
    builder.add_edge("retrieve_medical", "grade")
    builder.add_edge("grade", "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_edge("evaluate", "save")
    builder.add_edge("save", END)

    graph = builder.compile()
    return graph


# ----------
# Assembly
# ----------
def build_pipeline():
    embeddings, real_estate, medical = build_stores()
    llm = build_llm()
    query_rewriter = build_query_rewriter(llm)
    document_grader = build_document_grader(llm)
    question_router, real_estate_retriever, medical_retriever = build_router(llm, real_estate, medical)
    answer_chain = build_answer_chain(llm)
    evaluator = build_evaluator(llm)
    graph = build_graph(
        query_rewriter, question_router, real_estate_retriever,
        medical_retriever, document_grader, answer_chain, evaluator,
    )
    return graph
