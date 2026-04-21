"""
MainAgent -- RAG-powered support agent.

Provides two agent variants for regression testing:
  - MainAgent     : V1 baseline  (GPT-4o, simple prompt)
  - MainAgentV2   : V2 optimized (GPT-4.1)

"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# =========================================================
# DATA LOADING
# =========================================================

DB_FILE = Path(__file__).parent.parent / "data" / "vector_db.json"


def load_vector_store() -> List[Dict]:
    try:
        return json.loads(DB_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print("⚠️ Missing vector_db.json")
        return []


VECTOR_STORE = load_vector_store()
CHUNK_INDEX = {str(c["chunk_id"]): c for c in VECTOR_STORE}


# =========================================================
# TOKENIZATION + SCORING
# =========================================================

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def compute_bm25(query_tokens: List[str], doc_text: str) -> float:
    tokens = tokenize(doc_text)
    if not tokens:
        return 0.0

    k1, b = 1.5, 0.75
    avg_len = 120
    tf = {}

    for t in tokens:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    doc_len = len(tokens)

    for qt in set(query_tokens):
        freq = tf.get(qt, 0)
        if freq > 0:
            score += (freq * (k1 + 1)) / (
                freq + k1 * (1 - b + b * doc_len / avg_len)
            )

    return score


# =========================================================
# RETRIEVAL STRATEGIES
# =========================================================

def retrieve_dense(query: str, k: int) -> List[Dict]:
    tokens = tokenize(query)
    ranked = sorted(
        VECTOR_STORE,
        key=lambda c: compute_bm25(tokens, c["text"]),
        reverse=True,
    )
    return ranked[:k]


def retrieve_keyword(query: str, k: int) -> List[Dict]:
    keywords = [t for t in tokenize(query) if len(t) >= 3]

    scored = []
    for c in VECTOR_STORE:
        text = c["text"].lower()
        hits = sum(1 for kw in keywords if kw in text)
        scored.append((hits, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def reciprocal_rank_fusion(queries: List[str], k: int) -> List[Dict]:
    scores: Dict[str, float] = {}

    for q in queries:
        dense = retrieve_dense(q, 15)
        sparse = retrieve_keyword(q, 15)

        for rank, c in enumerate(dense):
            cid = str(c["chunk_id"])
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank + 1)

        for rank, c in enumerate(sparse):
            cid = str(c["chunk_id"])
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank + 1)

    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return [CHUNK_INDEX[cid] for cid in sorted_ids if cid in CHUNK_INDEX][:k]


def rerank_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    query_lower = query.lower()
    tokens = tokenize(query)

    scored = []
    for c in chunks:
        text = c["text"].lower()
        score = 0.0

        if query_lower in text:
            score += 2.0

        matches = sum(1 for t in tokens if t in text and len(t) > 2)
        score += matches * 0.5

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# =========================================================
# PROMPTS
# =========================================================

PROMPT_V1 = """Bạn là trợ lý AI. Trả lời dựa trên kiến thức và tài liệu."""

PROMPT_V2 = """Bạn là trợ lý nội bộ.

Chỉ trả lời dựa trên <context>.
Không dùng kiến thức ngoài.

Nếu không có thông tin:
"Không tìm thấy trong tài liệu."
"""


# =========================================================
# MAIN AGENT
# =========================================================

class MainAgent:
    def __init__(self, mode: str = "v1"):
        self.mode = mode.lower()
        self.model = "gpt-4o-mini"

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

        if not api_key:
            print(f"⚠️ Running in MOCK mode ({self.mode})")

    # -----------------------------
    # LLM CALL
    # -----------------------------
    async def _generate(self, system: str, user: str) -> Tuple[str, int]:
        if not self.client:
            return "Mock response (no API key)", 0

        try:
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            text = res.choices[0].message.content or ""
            tokens = res.usage.total_tokens if res.usage else 0
            return text, tokens

        except Exception as e:
            return f"LLM Error: {e}", 0

    # -----------------------------
    # PUBLIC ENTRY
    # -----------------------------
    async def query(self, question: str) -> Dict:
        if self.mode == "v2":
            return await self._query_v2(question)
        return await self._query_v1(question)

    # =========================================================
    # VERSION 1 (WEAK)
    # =========================================================
    async def _query_v1(self, question: str) -> Dict:
        docs = retrieve_dense(question, k=2)

        context = "\n".join(
            f"[{i+1}] {c['text']}" for i, c in enumerate(docs)
        )

        user_prompt = f"Tài liệu:\n{context}\n\nCâu hỏi: {question}"

        answer, tokens = await self._generate(PROMPT_V1, user_prompt)

        return {
            "answer": answer,
            "retrieved_ids": [str(c["chunk_id"]) for c in docs],
            "contexts": [c["text"] for c in docs],
            "metadata": {"mode": "v1", "tokens": tokens},
        }

    # =========================================================
    # VERSION 2 (IMPROVED)
    # =========================================================
    async def _query_v2(self, question: str) -> Dict:
        query_variants = [
            question,
            f"{question} chính sách",
            f"{question} hệ thống",
        ]

        candidates = reciprocal_rank_fusion(query_variants, k=15)
        ranked = rerank_chunks(question, candidates)
        top_chunks = ranked[:3]

        context = "\n\n".join(
            f"[{c['source']} - {c['section']}]\n{c['text']}"
            for c in top_chunks
        )

        user_prompt = f"<context>\n{context}\n</context>\n\nCâu hỏi: {question}"

        answer, tokens = await self._generate(PROMPT_V2, user_prompt)

        if "Không tìm thấy trong tài liệu" in answer:
            answer = "Không tìm thấy thông tin trong tài liệu."

        return {
            "answer": answer,
            "retrieved_ids": [str(c["chunk_id"]) for c in top_chunks],
            "contexts": [c["text"] for c in top_chunks],
            "metadata": {"mode": "v2", "tokens": tokens},
        }


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    async def test():
        q = "Thời gian tối đa để xử lý và khắc phục ticket P3 là gì?"
        agent1 = MainAgent("v1")
        print('v1', (await agent1.query(q))["answer"])
        agent2 = MainAgent("v2")
        print('v2', (await agent2.query(q))["answer"])

    asyncio.run(test())