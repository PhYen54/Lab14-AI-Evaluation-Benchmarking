import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent, MainAgentV2

from dotenv import load_dotenv
load_dotenv()


def _build_evaluator():
    try:
        return RetrievalEvaluator()
    except Exception:
        return None


def _build_judge():
    try:
        return LLMJudge()
    except ValueError:
        return None


def _regression_gate(v1_metrics: dict, v2_metrics: dict) -> dict:
    """
    Regression gate: compares V1 vs V2 and decides APPROVE / BLOCK / REVIEW.
    """
    v1_score = v1_metrics["avg_judge_score"]
    v2_score = v2_metrics["avg_judge_score"]
    score_delta = v2_score - v1_score

    cost_ratio = (
        v2_metrics["total_cost"] / v1_metrics["total_cost"]
        if v1_metrics["total_cost"] > 0 else 0.0
    )
    latency_delta = v2_metrics["avg_latency"] - v1_metrics["avg_latency"]

    if score_delta > 0.2 and cost_ratio <= 1.5:
        decision = "APPROVE"
        reason = "Significant score improvement within cost budget."
    elif score_delta < -0.2:
        decision = "BLOCK"
        reason = "Score regression detected."
    else:
        decision = "REVIEW"
        reason = "Marginal change -- manual review recommended."

    return {
        "decision": decision,
        "reason": reason,
        "v1_avg_score": round(v1_score, 4),
        "v2_avg_score": round(v2_score, 4),
        "score_delta": round(score_delta, 4),
        "cost_ratio": round(cost_ratio, 4),
        "latency_delta_seconds": round(latency_delta, 4),
    }


def _aggregate_metrics(results: list) -> dict:
    total = len(results)
    return {
        "total_cases": total,
        "total_cost": sum(r.get("cost_usd", 0) for r in results),
        "total_tokens": sum(r.get("tokens_used", 0) for r in results),
        "total_wall_clock_seconds": results[-1].get("_run_summary", {}).get("total_wall_clock_seconds", 0),
        "avg_judge_score": sum(r["judge"]["final_score"] for r in results) / total,
        "avg_faithfulness": sum(r["ragas"]["faithfulness"] for r in results) / total,
        "avg_relevancy": sum(r["ragas"]["relevancy"] for r in results) / total,
        "avg_hit_rate": sum(r["retrieval"]["hit_rate"] for r in results) / total,
        "avg_mrr": sum(r["retrieval"]["mrr"] for r in results) / total,
        "avg_agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        "avg_latency": sum(r["latency"] for r in results) / total,
        "pass_count": sum(1 for r in results if r["status"] == "pass"),
        "fail_count": sum(1 for r in results if r["status"] == "fail"),
    }


async def run_benchmark(agent_version: str, dataset: list, evaluator, judge, agent=None) -> tuple:
    print(f"\n[Runner] Starting benchmark for [{agent_version}] ({len(dataset)} cases, batch_size=5)...")

    if agent is None:
        agent = MainAgent()
    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)

    metrics = _aggregate_metrics(results)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS -- {agent_version}")
    print(f"{'='*60}")
    print(f"  Total cases  : {metrics['total_cases']}")
    print(f"  Pass / Fail  : {metrics['pass_count']} / {metrics['fail_count']}")
    print(f"  Avg Score    : {metrics['avg_judge_score']:.4f} / 5.0")
    print(f"  Hit Rate     : {metrics['avg_hit_rate']*100:.1f}%")
    print(f"  MRR          : {metrics['avg_mrr']*100:.1f}%")
    print(f"  Faithfulness : {metrics['avg_faithfulness']:.4f}")
    print(f"  Relevancy    : {metrics['avg_relevancy']:.4f}")
    print(f"  Agreemt Rate : {metrics['avg_agreement_rate']*100:.1f}%")
    print(f"  Avg Latency  : {metrics['avg_latency']:.3f}s")
    print(f"  Total Cost   : ${metrics['total_cost']:.6f}")
    print(f"  Total Tokens : {metrics['total_tokens']:,}")
    print(f"  Wall Clock   : {metrics['total_wall_clock_seconds']}s")
    print(f"{'='*60}")

    return results, metrics


async def main():
    # Load dataset
    dataset_path = "data/golden_set.jsonl"
    if not os.path.exists(dataset_path):
        print("[ERROR] data/golden_set.jsonl not found. Run M1's generator first.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("[ERROR] data/golden_set.jsonl is empty.")
        return

    print(f"[Dataset] Loaded {len(dataset)} test cases from {dataset_path}.")

    # Build components
    evaluator = _build_evaluator()
    judge = _build_judge()

    if judge is None:
        print("[ERROR] LLMJudge could not be initialized. Check OPENAI_API_KEY in .env.")
        return

    print(f"[OK] RetrievalEvaluator loaded: {evaluator is not None}")
    print(f"[OK] LLMJudge loaded: {judge.model_a} + {judge.model_b}")

    # Run V1 -- baseline agent (GPT-4o-mini)
    v1_agent = MainAgent()
    v1_results, v1_metrics = await run_benchmark("Agent_V1_Base", dataset, evaluator, judge, agent=v1_agent)

    # Run V2 -- optimized agent (GPT-4o + improved prompt)
    v2_agent = MainAgentV2()
    v2_results, v2_metrics = await run_benchmark("Agent_V2_Optimized", dataset, evaluator, judge, agent=v2_agent)

    # Regression gate
    gate = _regression_gate(v1_metrics, v2_metrics)

    print(f"\n{'='*60}")
    print(f"  REGRESSION ANALYSIS -- V1 vs V2")
    print(f"{'='*60}")
    print(f"  V1 Avg Score : {gate['v1_avg_score']}")
    print(f"  V2 Avg Score : {gate['v2_avg_score']}")
    print(f"  Score Delta  : {'+' if gate['score_delta'] >= 0 else ''}{gate['score_delta']}")
    print(f"  Cost Ratio   : {gate['cost_ratio']}x")
    print(f"  Latency Delta: {'+' if gate['latency_delta_seconds'] >= 0 else ''}{gate['latency_delta_seconds']}s")
    print(f"  GATE DECISION: [{gate['decision']}]")
    print(f"  Reason       : {gate['reason']}")
    print(f"{'='*60}")

    # Write reports
    os.makedirs("reports", exist_ok=True)

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_size": len(dataset),
        "agent_v1": "MainAgent (gpt-4o-mini)",
        "agent_v2": "MainAgentV2 (gpt-4o)",
        },
        "metrics": {
            "v1": v1_metrics,
            "v2": v2_metrics,
        },
        "regression": gate,
    }

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({"v1_results": v1_results, "v2_results": v2_results}, f, ensure_ascii=False, indent=2)

    print(f"\n[Reports] Written to:")
    print(f"   reports/summary.json          (V1/V2 comparison + regression gate)")
    print(f"   reports/benchmark_results.json ({len(v1_results)} case results per version)")


if __name__ == "__main__":
    asyncio.run(main())
