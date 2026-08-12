import csv
import re
import json
from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from using_langchain_langgraph import build_llm 


ground_truth_generator_instructions = """
You emulate a user of our retrieval system, which answers questions about
medical records/Medicare claims and real-estate listings, leases, and market data.
Formulate 5 questions this user might ask based on a single retrieved record.

The record comes from one of two domains:
- medical: a patient's clinical record or a Medicare claim - diagnoses,
  procedures, chronic conditions, billing/reimbursement amounts, dates of service.
- real_estate: anything about property or the housing market - listings, leases,
  location, price or rent, size, lease terms, features, availability, AND
  market-level data such as mortgage/interest rates, rental inventory or
  vacancy counts, and price or rate trends over time.

Treat mortgage rates, rental-listing counts, availability by month, and similar
market statistics as REAL ESTATE - never as out-of-domain.

First infer which domain the record belongs to, then write questions a real
person in that domain would plausibly ask: a patient, caregiver, or claims
analyst for medical; a renter, buyer, agent, or market analyst for real estate.

The record must contain the answer to every question, and the questions should
be complete and not too short. If possible, use as few words as possible from
the record.

The output should resemble how people actually ask questions on the internet.
Not too formal, not too short, not too long.
""".strip()

llm = build_llm()
ground_truth_prompt = ChatPromptTemplate.from_messages([
    ("system", ground_truth_generator_instructions),
    ("human", "{question}"),
])
ground_truth_generator = ground_truth_prompt | llm | StrOutputParser()


# ---- config ----
INPUT_DIR = Path("./query_results")        # folder holding the .json files
OUTPUT_PATH = Path("./data/ground_truth.csv") # where results are written


def extract_questions(obj):
    """Pull question strings out of a loaded JSON object, tolerating a few shapes."""
    if isinstance(obj, dict):
        if isinstance(obj.get("question"), str):
            return [obj["question"]]
        if isinstance(obj.get("questions"), list):
            return [q for q in obj["questions"] if isinstance(q, str)]
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(extract_questions(item))
        return out
    return []


def load_questions(input_dir=INPUT_DIR):
    items = []
    for path in sorted(Path(input_dir).glob("*.json")):   # use rglob for subfolders
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  [skip] {path.name}: invalid JSON ({e})")
            continue
        for q in extract_questions(data):
            items.append({"source_file": path.name, "question": q})
    return items


def split_questions(text):
    """Break the generator's output (a numbered/bulleted list) into individual questions."""
    if not text:
        return []
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading "1.", "2)", "-", "*", "Q1:" etc.
        cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*•]|Q\d*[:.]?)\s*", "", line).strip()
        if cleaned:
            questions.append(cleaned)
    return questions


def generate_ground_truth(input_dir=INPUT_DIR, output_path=OUTPUT_PATH):
    items = load_questions(input_dir)
    print(f"Found {len(items)} questions in {input_dir}")

    fieldnames = ["source_file", "question", "generated_question"]
    results = []
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(items, 1):
            question = item["question"]
            try:
                generated_data = ground_truth_generator.invoke({"question": question})
            except Exception as e:
                print(f"  [error] q{i}: {e}")
                continue

            for gq in split_questions(generated_data):
                record = {
                    "source_file": item["source_file"],
                    "question": question,
                    "generated_question": gq,
                }
                results.append(record)
                writer.writerow(record)

            print(f"  [{i}/{len(items)}] {question[:60]}")

    #print(f"\nSaved {len(results)} rows to {output_path.resolve()}")
    return results


###################################################################################
########################### Evaluating Ground Truth ###############################
###################################################################################

# --- structured similarity judge ---
class SimilarityCheck(BaseModel):
    """Whether a generated question asks for the same information as the source."""

    similar: Literal["yes", "no"] = Field(
        ..., description="Do both questions seek the same information? 'yes' or 'no'."
    )
    similarity_score: int = Field(
        ..., ge=1, le=5,
        description="1 = unrelated, 5 = same intent / paraphrase.",
    )
    rationale: str = Field(..., description="One sentence explaining the verdict.")


sim_system = (
    "You judge whether two questions ask for the same information. Focus on "
    "intent and the answer they'd require, not wording - a casual paraphrase of "
    "the same question is 'yes'. Different topic, scope, or required answer is "
    "'no'. Be consistent."
)
sim_prompt = ChatPromptTemplate.from_messages([
    ("system", sim_system),
    ("human", "Source question:\n{question}\n\nGenerated question:\n{generated_question}"),
])
similarity_checker = sim_prompt | llm.with_structured_output(SimilarityCheck, method="function_calling")


def load_records(path="./data/ground_truth.csv"):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check_similarity(records=None, in_path="./data/ground_truth.csv", output_path="./data/ground_truth_checked.csv"):
    records = records if records is not None else load_records(in_path)

    fieldnames = ["source_file", "question", "generated_question",
                  "similar", "similarity_score", "rationale"]
    checked = []
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(records, 1):
            verdict = similarity_checker.invoke({
                "question": r["question"],
                "generated_question": r["generated_question"],
            })
            if not isinstance(verdict, SimilarityCheck):   # guard: raw message
                verdict = SimilarityCheck.model_validate_json(verdict.content)

            row = {
                "source_file": r["source_file"],
                "question": r["question"],
                "generated_question": r["generated_question"],
                "similar": verdict.similar,
                "similarity_score": verdict.similarity_score,
                "rationale": verdict.rationale,
            }
            checked.append(row)
            writer.writerow(row)
            print(f"  [{i}/{len(records)}] {verdict.similar} ({verdict.similarity_score}/5)")

    passed = sum(1 for r in checked if r["similar"] == "yes")
    print(f"\n{passed}/{len(checked)} generated questions matched the source "
          f"({passed / len(checked):.0%}). Saved to {output_path}")
    return checked


#####################################################################################################
################################### ------------------------- #######################################
#####################################################################################################

