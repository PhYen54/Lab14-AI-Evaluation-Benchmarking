import asyncio
import time
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cost estimation (per 1K tokens) - adjust based on model
COST_PER_1K_TOKENS = 0.00015

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.total_tokens = 0
        self.total_cost = 0.0

    def _estimate_cost(self, tokens_used: int) -> float:
        """Estimate cost based on token usage (USD)."""
        return round((tokens_used / 1000) * COST_PER_1K_TOKENS, 6)

    async def run_single_test(self, test_case: Dict) -> Dict:
        """Run single benchmark test with full tracking: latency, tokens, cost, errors."""
        wall_start = time.perf_counter()
        
        try:
            # 1. Gọi Agent (with latency tracking)
            agent_start = time.perf_counter()
            response = await self.agent.query(test_case["question"])
            agent_latency = time.perf_counter() - agent_start
            
            # Extract tokens_used from response or estimate
            tokens_used = response.get("metadata", {}).get("tokens_used", 0)
            if not tokens_used:
                # Rough estimate: ~4 chars per token
                tokens_used = max(len(test_case["question"]) // 4, 10) + max(len(str(response.get("answer", ""))) // 4, 10)
            
            cost_usd = self._estimate_cost(tokens_used)
            
            # 2. Chạy RAGAS metrics (Retrieval Evaluation)
            retrieval_start = time.perf_counter()
            ragas_scores = await self.evaluator.score(test_case, response)
            retrieval_latency = time.perf_counter() - retrieval_start
            
            # 3. Chạy Multi-Judge
            judge_start = time.perf_counter()
            judge_result = await self.judge.evaluate_multi_judge(
                test_case["question"], 
                response.get("answer", ""), 
                test_case.get("expected_answer", "")
            )
            judge_latency = time.perf_counter() - judge_start
            
            total_latency = time.perf_counter() - wall_start
            
            # Track cumulative
            self.total_tokens += tokens_used
            self.total_cost += cost_usd
            
            result = {
                "test_case": test_case["question"],
                "agent_response": response.get("answer", ""),
                "contexts": response.get("contexts", []),
                "latency": round(total_latency, 4),
                "agent_latency": round(agent_latency, 4),
                "retrieval_latency": round(retrieval_latency, 6),
                "judge_latency": round(judge_latency, 4),
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "ragas": ragas_scores,
                "judge": judge_result,
                "status": "pass" if judge_result.get("final_score", 1.0) >= 3.0 else "fail"
            }
            
            logger.info(f"✓ {test_case['question'][:40]}... | Score: {judge_result.get('final_score')} | Cost: ${cost_usd:.6f}")
            return result
            
        except Exception as e:
            logger.error(f"✗ Test failed: {test_case.get('question', 'N/A')[:40]}... | Error: {str(e)}")
            # Return error result instead of crashing
            return {
                "test_case": test_case.get("question", "UNKNOWN"),
                "agent_response": "",
                "contexts": [],
                "latency": round(time.perf_counter() - wall_start, 4),
                "agent_latency": 0.0,
                "retrieval_latency": 0.0,
                "judge_latency": 0.0,
                "tokens_used": 0,
                "cost_usd": 0.0,
                "ragas": {"hit_rate": 0.0, "mrr": 0.0, "expected_ids": [], "retrieved_ids": []},
                "judge": {"final_score": 1.0, "agreement_rate": 0.0, "cohens_kappa": 0.0, "individual_scores": {}},
                "status": "fail",
                "error": str(e)
            }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Run all benchmarks with async batching and detailed tracking.
        Returns: (results, summary_stats)
        """
        logger.info(f"🚀 Starting benchmark for {len(dataset)} cases with batch_size={batch_size}...")
        
        results = []
        batch_count = 0
        
        for i in range(0, len(dataset), batch_size):
            batch_count += 1
            batch = dataset[i:i + batch_size]
            logger.info(f"📦 Batch {batch_count}: {len(batch)} cases...")
            
            tasks = [self.run_single_test(case) for case in batch]
            # Use return_exceptions to handle batch failures gracefully
            batch_results = await asyncio.gather(*tasks, return_exceptions=False)
            results.extend(batch_results)
        
        summary = self._compute_summary(results, len(dataset), batch_size, batch_count)
        logger.info(f"✅ Complete. Pass: {summary['pass_count']} | Fail: {summary['fail_count']} | Cost: ${summary['total_cost_usd']:.6f}")
        
        return results, summary
    
    def _compute_summary(self, results: List[Dict], total_dataset_size: int, batch_size: int, batch_count: int) -> Dict:
        """Compute comprehensive summary statistics."""
        pass_count = sum(1 for r in results if r.get("status") == "pass")
        fail_count = len(results) - pass_count
        
        total_latency = sum(r.get("latency", 0) for r in results)
        avg_latency = total_latency / len(results) if results else 0
        
        return {
            "total_cases": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(100 * pass_count / len(results), 2) if results else 0.0,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "avg_cost_per_case": round(self.total_cost / len(results), 6) if results else 0.0,
            "total_wall_clock_seconds": round(total_latency, 2),
            "avg_latency_seconds": round(avg_latency, 4),
            "batch_count": batch_count,
            "batch_size": batch_size
        }