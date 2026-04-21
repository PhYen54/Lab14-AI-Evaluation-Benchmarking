import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DOCS_PATH = Path('data')
VECTOR_OUT = Path(__file__).parent / "vector_db.json"
GOLDEN_OUT = Path(__file__).parent / "golden_set.jsonl"

TARGET_SIZE = 50

# =========================
# VECTOR DB BUILDING
# =========================

def extract_chunks() -> List[Dict]:
    chunks: List[Dict] = []

    for file in sorted(DOCS_PATH.glob("*.txt")):
        content = file.read_text(encoding="utf-8")
        lines = content.splitlines()

        meta = {"source": "", "department": ""}
        for ln in lines[:8]:
            if ln.startswith("Source:"):
                meta["source"] = ln[7:].strip()
            elif ln.startswith("Department:"):
                meta["department"] = ln[11:].strip()

        sections = re.split(r"===\s*(.+?)\s*===", content)

        for i in range(1, len(sections) - 1, 2):
            section_name = sections[i].strip()
            body = sections[i + 1].strip()

            if not body:
                continue

            chunks.append({
                "chunk_id": f"{file.stem}_c{len(chunks)+1}",
                "source": meta["source"],
                "section": section_name,
                "text": body
            })

    return chunks


# =========================
# PROMPT BUILDER
# =========================

def build_generation_prompt(chunks: List[Dict], distribution: Dict[str, int]) -> str:
    payload = json.dumps(
        [{"id": c["chunk_id"], "txt": c["text"][:500]} for c in chunks],
        ensure_ascii=False
    )

    total = sum(distribution.values())

    return f"""
Create {total} Vietnamese RAG test cases using the following chunks:
{payload}

Distribution:
{json.dumps(distribution)}

Each item MUST include:
- question
- expected_answer
- ground_truth_id
- expected_retrieval_ids (list)
- difficulty (easy|medium|hard|multi_hop)
- type (standard|adversarial|edge_case)
- category

Return ONLY a JSON array.
"""


# =========================
# LLM CALL
# =========================

async def generate_cases(prompt: str) -> str:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content


# =========================
# PARSER
# =========================

def safe_parse_json(raw: str) -> List[Dict]:
    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    try:
        data = json.loads(raw)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["cases", "data", "test_cases"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
    except:
        pass

    return []


# =========================
# BATCH CONFIG
# =========================

BATCHES = [
    ("SLA", ["sla_p1_2026"], {"easy": 4, "medium": 4, "hard": 2, "multi_hop": 2, "adversarial": 1}),
    ("Refund", ["policy_refund_v4"], {"easy": 4, "medium": 3, "hard": 2, "multi_hop": 2, "adversarial": 2}),
    ("Access", ["access_control_sop"], {"easy": 4, "medium": 3, "hard": 2, "multi_hop": 2, "adversarial": 1}),
    ("HR", ["hr_leave_policy"], {"easy": 3, "medium": 3, "hard": 2, "multi_hop": 2, "adversarial": 2}),
    ("IT", ["it_helpdesk_faq"], {"easy": 3, "medium": 3, "hard": 2, "multi_hop": 2, "adversarial": 0}),
    ("Cross", ["sla", "access", "refund", "hr", "it"], {"medium": 2, "hard": 3, "multi_hop": 3, "adversarial": 2}),
]


# =========================
# EXECUTION (NO MAIN)
# =========================

print("🔧 Building vector database...")
vector_db = extract_chunks()
valid_ids = {c["chunk_id"] for c in vector_db}

print(f"Loaded {len(vector_db)} chunks")

all_samples: List[Dict] = []


async def run_generation():
    for name, patterns, dist in BATCHES:
        print(f"\n📦 Processing batch: {name}")

        batch_chunks = [
            c for c in vector_db
            if any(p in c["chunk_id"].lower() for p in patterns)
        ]

        if not batch_chunks:
            print("  → Skipped (no data)")
            continue

        prompt = build_generation_prompt(batch_chunks, dist)
        raw_output = await generate_cases(prompt)

        cases = safe_parse_json(raw_output)
        print(f"  → Generated: {len(cases)}")

        valid_cases = [
            c for c in cases
            if c.get("ground_truth_id") in valid_ids
        ]

        print(f"  → Valid: {len(valid_cases)}")

        all_samples.extend(valid_cases)


asyncio.run(run_generation())


# =========================
# SAVE RESULTS
# =========================

print(f"\n💾 Saving {len(all_samples)} samples...")

with open(GOLDEN_OUT, "w", encoding="utf-8") as f:
    for item in all_samples[:TARGET_SIZE]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(VECTOR_OUT, "w", encoding="utf-8") as f:
    json.dump(vector_db, f, ensure_ascii=False, indent=2)

print(f"✅ Done. Saved to {GOLDEN_OUT}")