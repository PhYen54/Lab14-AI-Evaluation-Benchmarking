import asyncio
import time
from typing import List, Dict, Optional


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        overall_start = time.perf_counter()
        retrieval_start = 0.0
        judge_start = 0.0

        # 1. Call Agent
        response_start = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        agent_latency = time.perf_counter() - response_start

        # Extract token usage from agent response
        metadata = response.get("metadata", {})
        tokens_used = metadata.get("tokens_used", 150)
        model_name = metadata.get("model", "unknown")

        # 2. Calculate cost based on model
        cost_usd = self._calculate_cost(tokens_used, model_name)

        # 3. Run Retrieval Evaluator (Hit Rate + MRR)
        retrieval_start = time.perf_counter()
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])

        hit_rate = self.evaluator.calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)
        mrr = self.evaluator.calculate_mrr(expected_ids, retrieved_ids)
        retrieval_latency = time.perf_counter() - retrieval_start

        retrieval_metrics = {
            "hit_rate": hit_rate,
            "mrr": mrr,
            "latency": retrieval_latency,
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
        }

        # 4. Run RAGAS-style metrics
        ragas_scores = {
            "faithfulness": self._estimate_faithfulness(response),
            "relevancy": self._estimate_relevancy(test_case, response),
            "latency": 0.0,
        }

        # 5. Run Multi-Judge
        judge_start = time.perf_counter()
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case.get("expected_answer", ""),
        )
        judge_latency = time.perf_counter() - judge_start

        # Attach judge latency
        judge_result["latency"] = judge_latency

        overall_latency = time.perf_counter() - overall_start

        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "contexts": response.get("contexts", []),
            "latency": overall_latency,
            "agent_latency": agent_latency,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "ragas": ragas_scores,
            "retrieval": retrieval_metrics,
            "judge": judge_result,
            "status": "pass" if judge_result.get("final_score", 0) >= 3 else "fail",
        }

    async def run_all(
        self, dataset: List[Dict], batch_size: int = 5
    ) -> List[Dict]:
        """
        Run tests in parallel using asyncio.gather() with batch_size limit
        to avoid hitting API rate limits.
        Tracks total wall-clock time, accumulated cost, and progress.
        """
        total_cases = len(dataset)
        total_batches = (total_cases + batch_size - 1) // batch_size
        results = []
        total_cost = 0.0
        total_tokens = 0
        total_latency = 0.0

        overall_start = time.perf_counter()

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_cases)
            batch = dataset[batch_start:batch_end]

            print(
                f"  Batch {batch_idx + 1}/{total_batches}: "
                f"Running cases {batch_start + 1}–{batch_end}..."
            )

            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Accumulate stats for this batch
            for r in batch_results:
                total_cost += r.get("cost_usd", 0.0)
                total_tokens += r.get("tokens_used", 0)
                total_latency += r.get("latency", 0.0)

            print(
                f"  Batch {batch_idx + 1}/{total_batches} done. "
                f"Cumulative cost so far: ${total_cost:.4f}"
            )

        overall_elapsed = time.perf_counter() - overall_start

        # Attach aggregate summary to each result (last one holds the summary)
        summary = {
            "total_cases": total_cases,
            "total_batches": total_batches,
            "batch_size": batch_size,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "total_wall_clock_seconds": round(overall_elapsed, 2),
            "total_cpu_seconds": round(total_latency, 2),
        }
        for r in results:
            r["_run_summary"] = summary

        print(
            f"\n  Benchmark complete: {total_cases} cases in "
            f"{overall_elapsed:.2f}s | Total cost: ${total_cost:.4f} | "
            f"Total tokens: {total_tokens:,}"
        )

        return results

    def _calculate_cost(self, tokens_used: int, model_name: str = "unknown") -> float:
        """
        Estimate API cost in USD based on token usage.
        Pricing is per 1M tokens (input + output combined approximation).
        """
        # Per-1M-token pricing (input + output blended)
        pricing = {
            "gpt-4o": 2.50,
            "gpt-4o-mini": 0.15,
            "gpt-4-turbo": 10.00,
            "gpt-3.5-turbo": 0.50,
            "claude-3-5-sonnet": 3.00,
            "claude-3-5-haiku": 0.25,
            "claude-3-opus": 15.00,
            "gemini-1.5-pro": 1.25,
            "gemini-1.5-flash": 0.075,
            "unknown": 1.00,
        }
        rate = pricing.get(model_name.lower(), pricing["unknown"])
        return (tokens_used / 1_000_000) * rate

    def _estimate_faithfulness(self, response: Dict) -> float:
        """
        Estimate faithfulness: does the answer stick to the retrieved contexts?
        Uses simple heuristic: check overlap between answer and contexts.
        Replace with real RAGAS faithfulness metric in production.
        """
        answer = response.get("answer", "").lower()
        contexts = " ".join(response.get("contexts", [])).lower()

        if not answer or not contexts:
            return 0.0

        # Simple word-overlap heuristic
        answer_words = set(answer.split())
        context_words = set(contexts.split())
        overlap = len(answer_words & context_words)
        ratio = overlap / max(len(answer_words), 1)

        # Penalise hallucinated claims (boilerplate phrases that aren't in context)
        generic_phrases = [
            "i don't know", "as an ai", "i cannot", "unable to",
            "not sure", "i'm not sure", "based on my knowledge",
        ]
        for phrase in generic_phrases:
            if phrase in answer:
                ratio *= 0.9

        return round(min(ratio, 1.0), 3)

    def _estimate_relevancy(self, test_case: Dict, response: Dict) -> float:
        """
        Estimate answer relevancy: does the answer address the question?
        Uses keyword overlap between question and answer.
        Replace with real RAGAS relevancy metric in production.
        """
        question = test_case.get("question", "").lower()
        answer = response.get("answer", "").lower()

        if not question or not answer:
            return 0.0

        q_words = set(question.split())
        a_words = set(answer.split())

        # Remove stop words for fairer comparison
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "of", "to",
            "in", "on", "at", "for", "how", "what", "when", "where",
            "why", "can", "you", "i", "my", "me", "and", "or", "be",
            "do", "does", "did", "that", "this", "with", "from",
        }
        q_meaningful = q_words - stop_words
        a_meaningful = a_words - stop_words

        if not q_meaningful:
            return 0.0

        overlap = len(q_meaningful & a_meaningful)
        return round(overlap / len(q_meaningful), 3)
