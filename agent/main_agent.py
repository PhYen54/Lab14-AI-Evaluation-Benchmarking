"""
MainAgent -- RAG-powered support agent.

Provides two agent variants for regression testing:
  - MainAgent     : V1 baseline  (GPT-4o-mini, simple prompt)
  - MainAgentV2   : V2 optimized (GPT-4o,  improved prompt + retrieval)

Each agent implements:
  - query(question: str) -> {
      "answer": str,
      "contexts": List[str],          # retrieved chunks
      "metadata": {
          "model": str,               # model used
          "tokens_used": int,          # input + output tokens
          "retrieved_ids": List[str],  # chunk IDs (simulated)
      }
    }
"""

import asyncio
import hashlib
import re
from typing import Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Shared: naive keyword-based retrieval over a corpus
# In production, replace with a real vector DB (FAISS / Chroma / Pinecone).
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "là", "của", "được", "để", "trong", "và", "có", "không", "cho",
    "với", "the", "a", "an", "is", "are", "was", "to", "of", "and",
    "in", "on", "at", "for", "how", "what", "when", "where", "why",
    "can", "you", "i", "my", "me", "be", "do", "does", "that", "this",
}


def _content_corpus() -> Dict[str, str]:
    """
    Returns a small in-memory corpus used for retrieval simulation.
    In a real system this would be loaded from a vector store.
    Maps chunk_id -> chunk_text.
    """
    return {
        "chunk_001": (
            "AI Evaluation là một quy trình kỹ thuật nhằm đo lường chất lượng "
            "của các hệ thống AI. Quy trình này bao gồm việc đánh giá độ chính xác, "
            "sự an toàn, và khả năng chịu tải của mô hình."
        ),
        "chunk_002": (
            "AI Evaluation có thể được thực hiện thông qua các phương pháp thủ công "
            "hoặc tự động. Các công cụ phổ biến bao gồm RAGAS, Trulens, và BLEU/ROUGE "
            "cho các bài toán generattion."
        ),
        "chunk_003": (
            "Mục tiêu chính của AI Evaluation là đảm bảo các hệ thống AI hoạt động "
            "đúng như dự kiến, an toàn, và đáng tin cậy trước khi triển khai sản phẩm."
        ),
        "chunk_004": (
            "Hit Rate là tỷ lệ các câu hỏi mà hệ thống RAG trả lời đúng. "
            "MRR (Mean Reciprocal Rank) đo thứ hạng trung bình của câu trả lời đúng đầu tiên."
        ),
        "chunk_005": (
            "AI Evaluation không chỉ dành cho các chuyên gia kỹ thuật mà còn cho "
            "các nhà quản lý và người dùng cuối muốn xác minh chất lượng sản phẩm AI."
        ),
        "chunk_006": (
            "Multi-Judge là phương pháp sử dụng nhiều mô hình AI (ví dụ GPT-4o và Claude) "
            "để chấm điểm cùng một câu trả lời, giúp tăng độ tin cậy và giảm thiên lệch "
            "từ một model duy nhất."
        ),
        "chunk_007": (
            "Regression Gate là một cơ chế tự động so sánh phiên bản mới của Agent "
            "với phiên bản cũ. Nếu điểm số giảm quá ngưỡng, hệ thống sẽ tự động "
            "BLOCK bản cập nhật để tránh regression."
        ),
        "chunk_008": (
            "Faithfulness trong RAGAS đo lường mức độ câu trả lời bám sát vào ngữ cảnh "
            "được trích xuất, không bịa đặt thông tin (hallucination)."
        ),
        "chunk_009": (
            "Relevancy đánh giá mức độ câu trả lời phù hợp với câu hỏi được đặt ra. "
            "Một câu trả lời có thể đúng nhưng không liên quan nếu nó không giải quyết "
            "đúng vấn đề người dùng hỏi."
        ),
        "chunk_010": (
            "AI Evaluation cần được tích hợp vào quy trình CI/CD để đảm bảo mọi bản cập "
            "nhật đều được kiểm tra tự động trước khi release ra production."
        ),
    }


def _keyword_retrieval(question: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Naive keyword-based retrieval: returns top_k chunks most relevant to the question.
    Replaced by a real vector store in production.
    """
    corpus = _content_corpus()
    q_words = set(
        w.lower().strip(".,!?;:\"'()[]{}")
        for w in question.split()
        if len(w) > 2 and w.lower() not in STOP_WORDS
    )

    if not q_words:
        q_words = {question.lower()}

    scored = []
    for chunk_id, text in corpus.items():
        t_words = set(
            w.lower().strip(".,!?;:\"'()[]{}")
            for w in text.split()
            if len(w) > 2
        )
        overlap = q_words & t_words
        if overlap:
            scored.append((len(overlap), chunk_id, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"chunk_id": chunk_id, "text": text, "score": score}
        for score, chunk_id, text in scored[:top_k]
    ]


def _chunk_id_from_text(text: str) -> str:
    """Generate a stable chunk ID from text content."""
    return "chunk_" + hashlib.md5(text.encode()).hexdigest()[:6]


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Async RAG agent with OpenAI backend.
    Subclasses override _build_prompt() to customise generation behaviour.
    """

    def __init__(self, model: str, name: str):
        self.model = model
        self.name = name
        self.client = AsyncOpenAI()

    def _build_system_prompt(self) -> str:
        raise NotImplementedError

    def _build_user_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = "\n\n---\n".join(
            f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)
        )
        return (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{context_block}\n\n"
            f"Hãy trả lời dựa trên ngữ cảnh trên. "
            f"Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ rằng bạn "
            f"không tìm thấy thông tin đó trong tài liệu."
        )

    async def query(self, question: str) -> Dict:
        """
        Execute one RAG cycle:
          1. Retrieve relevant context chunks.
          2. Build a prompt with context + question.
          3. Call the LLM and track token usage.
          4. Return the structured response.
        """
        # Step 1 -- Retrieval
        retrieved = _keyword_retrieval(question, top_k=3)
        if not retrieved:
            retrieved = [
                {"chunk_id": "chunk_default", "text": "Khong co ngữ cảnh phù hợp.", "score": 0}
            ]

        contexts = [r["text"] for r in retrieved]
        retrieved_ids = [r["chunk_id"] for r in retrieved]

        # Step 2 -- Count input tokens (rough estimate: ~4 chars per token)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question, contexts)
        input_text = system_prompt + user_prompt
        input_tokens = max(len(input_text) // 4, 10)

        # Step 3 -- LLM call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        answer = response.choices[0].message.content or ""
        usage = response.usage
        output_tokens = usage.completion_tokens if usage else max(len(answer) // 4, 10)
        total_tokens = input_tokens + output_tokens

        return {
            "answer": answer.strip(),
            "contexts": contexts,
            "metadata": {
                "model": self.model,
                "tokens_used": total_tokens,
                "retrieved_ids": retrieved_ids,
                "sources": ["knowledge_base"],
                "retrieval_scores": [r["score"] for r in retrieved],
            },
        }


# ---------------------------------------------------------------------------
# V1 -- Baseline agent
# ---------------------------------------------------------------------------

class MainAgent(BaseAgent):
    """
    V1 Baseline RAG Agent.
    - Model:     GPT-4o-mini (cheaper, acceptable quality)
    - Strategy:  Simple keyword retrieval + straightforward prompt
    """

    def __init__(self):
        super().__init__(model="gpt-4o-mini", name="SupportAgent-v1")

    def _build_system_prompt(self) -> str:
        return (
            "Bạn là một trợ lý AI hỗ trợ khách hàng. "
            "Trả lời ngắn gọn, chính xác, và chuyên nghiệp. "
            "Chỉ dựa vào ngữ cảnh được cung cấp. "
            "Nếu không biết, hãy nói 'Tôi không tìm thấy thông tin đó trong tài liệu.'"
        )


# ---------------------------------------------------------------------------
# V2 -- Optimized agent
# ---------------------------------------------------------------------------

class MainAgentV2(BaseAgent):
    """
    V2 Optimized RAG Agent.
    - Model:     GPT-4o (higher quality)
    - Strategy:  Improved prompt with safety + citation + out-of-scope guard
    """

    def __init__(self):
        super().__init__(model="gpt-4o", name="SupportAgent-v2")

    def _build_system_prompt(self) -> str:
        return (
            "Bạn là một chuyên gia AI tư vấn kỹ thuật. "
            "Trả lời CHÍNH XÁC, ĐẦY ĐỦ, và AN TOÀN dựa trên ngữ cảnh. "
            "Luôn ghi rõ nguồn từ ngữ cảnh nào bạn lấy thông tin. "
            "Nếu câu hỏi nằm ngoài ngữ cảnh, hãy trả lời: "
            "'Thông tin này không có trong tài liệu được cung cấp.'"
            "Không được bịa đặt hoặc suy đoán."
        )

    def _build_user_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = "\n\n---\n".join(
            f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)
        )
        return (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{context_block}\n\n"
            f"Hãy trả lời dựa trên ngữ cảnh trên. "
            f"Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ rằng bạn "
            f"không tìm thấy thông tin đó trong tài liệu.\n\n"
            f"LUÔN ghi rõ bạn lấy thông tin từ Context nào."
        )


if __name__ == "__main__":
    async def _test():
        agent = MainAgent()
        result = await agent.query("AI Evaluation là gì?")
        print("=== V1 ===")
        print(result["answer"])
        print(f"Tokens: {result['metadata']['tokens_used']}")

        agent2 = MainAgentV2()
        result2 = await agent2.query("AI Evaluation là gì?")
        print("\n=== V2 ===")
        print(result2["answer"])
        print(f"Tokens: {result2['metadata']['tokens_used']}")

    asyncio.run(_test())
