import csv
import re
import json
from pathlib import Path
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
