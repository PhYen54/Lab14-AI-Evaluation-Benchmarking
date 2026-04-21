import asyncio
import json
import os
import time
from typing import List, Tuple, Optional, Dict, Any

from engine.runner import BenchmarkRunner
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge

DATA_PATH = "data/golden_set.jsonl"
REPORT_DIR = "reports"


# =========================================================
# LOAD DATA
# =========================================================
def load_dataset(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        print(f" Run 'python data/synthetic_gen.py' first.")
        return None

    with open(path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print(f"{path} is empty. ")
        return None

    return dataset


# =========================================================
# METRICS - Thém chi phí, tokens, latency
# =========================================================
def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute metrics dictionary from results (without metadata wrapper).
    Used for both V1 and V2, then merged into single summary.
    """
    total = len(results)

    if total == 0:
        return {
            "total_cases": 0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_wall_clock_seconds": 0.0,
            "avg_judge_score": 0.0,
            "avg_faithfulness": 0.0,
            "avg_relevancy": 0.0,
            "avg_hit_rate": 0.0,
            "avg_mrr": 0.0,
            "avg_agreement_rate": 0.0,
            "avg_latency": 0.0,
            "pass_count": 0,
            "fail_count": 0
        }

    # Calculate metrics
    pass_count = sum(1 for r in results if r.get("status") == "pass")
    fail_count = total - pass_count
    
    avg_judge_score = sum(r["judge"].get("final_score", 1) for r in results) / total
    avg_faithfulness = sum(r["ragas"].get("faithfulness", 0) for r in results) / total
    avg_relevancy = sum(r["ragas"].get("relevancy", 0) for r in results) / total
    
    # Handle both flat and nested ragas structure
    avg_hit_rate = 0.0
    avg_mrr = 0.0
    for r in results:
        ragas = r.get("ragas", {})
        if isinstance(ragas, dict):
            if "hit_rate" in ragas:
                avg_hit_rate += ragas["hit_rate"]
            elif "retrieval" in ragas:
                avg_hit_rate += ragas["retrieval"].get("hit_rate", 0)
            
            if "mrr" in ragas:
                avg_mrr += ragas["mrr"]
            elif "retrieval" in ragas:
                avg_mrr += ragas["retrieval"].get("mrr", 0)
    avg_hit_rate /= total
    avg_mrr /= total
    
    avg_agreement_rate = sum(r["judge"].get("agreement_rate", 0) for r in results) / total
    
    total_tokens = sum(r.get("tokens_used", 0) for r in results)
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_latency = sum(r.get("latency", 0) for r in results)
    avg_latency = total_latency / total if total > 0 else 0

    pass_rate = (pass_count / total * 100) if total > 0 else 0.0

    return {
        "total_cases": total,
        "total_cost": round(total_cost, 7),
        "total_tokens": total_tokens,
        "total_wall_clock_seconds": round(total_latency, 2),
        "avg_judge_score": round(avg_judge_score, 5),
        "avg_score": round(avg_judge_score, 5),
        "avg_faithfulness": round(avg_faithfulness, 5),
        "avg_relevancy": round(avg_relevancy, 5),
        "avg_hit_rate": round(avg_hit_rate, 4),
        "hit_rate": round(avg_hit_rate, 4),
        "avg_mrr": round(avg_mrr, 4),
        "avg_agreement_rate": round(avg_agreement_rate, 5),
        "agreement_rate": round(avg_agreement_rate, 5),
        "avg_latency": avg_latency,
        "pass_rate": round(pass_rate, 2),
        "pass_count": pass_count,
        "fail_count": fail_count
    }


def compute_summary(agent_version: str, results: List[Dict], summary_dict: Optional[Dict] = None) -> Dict:
    """
    Compute comprehensive summary with cost, tokens, latency breakdown.
    For backward compatibility with existing code.
    """
    metrics = compute_metrics(results)
    
    return {
        "metadata": {
            "version": agent_version,
            "total": metrics["total_cases"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": metrics
    }


# =========================================================
# CORE RUNNER
# =========================================================
def resolve_agent_mode(agent_version: str) -> str:
    """
    Map tên version -> mode đúng của agent
    """
    return "v2" if "Optimized" in agent_version else "v1"


async def run_benchmark_with_results(
    agent_version: str,
    dataset: List[Dict]
) -> Tuple[List[Dict], Dict]:

    print(f"Running Benchmark for {agent_version}...")

    agent_mode = resolve_agent_mode(agent_version)

    agent = MainAgent(mode=agent_mode)

    runner = BenchmarkRunner(
        agent,
        RetrievalEvaluator(),
        LLMJudge()
    )

    # runner.run_all() now returns (results, summary_dict)
    results, runner_summary = await runner.run_all(dataset)
    summary = compute_summary(agent_version, results, runner_summary)

    return results, summary


async def run_single(agent_version: str, dataset: List[Dict]) -> Dict:
    _, summary = await run_benchmark_with_results(agent_version, dataset)
    return summary


# =========================================================
# REGRESSION GATE & COST ANALYSIS (Thành viên 5)
# =========================================================
def print_comparison(v1: Dict, v2: Dict) -> Tuple[str, Dict]:
    """
    Compare V1 vs V2 with cost, latency, quality delta.
    Returns: (decision, regression_metrics)
    """
    print("\n📊 --- REGRESSION COMPARISON (V1 vs V2) ---")

    v1_metrics = v1["metrics"]
    v2_metrics = v2["metrics"]
    
    # Quality metrics
    v1_score = v1_metrics.get("avg_judge_score", 0)
    v2_score = v2_metrics.get("avg_judge_score", 0)
    score_delta = v2_score - v1_score
    
    # Cost metrics
    v1_cost = v1_metrics.get("total_cost", 0)
    v2_cost = v2_metrics.get("total_cost", 0)
    cost_ratio = v2_cost / v1_cost if v1_cost > 0 else 1.0
    cost_delta = v2_cost - v1_cost
    cost_increase_pct = round(100 * (cost_ratio - 1), 2)
    
    # Latency metrics
    v1_latency = v1_metrics.get("avg_latency", 0)
    v2_latency = v2_metrics.get("avg_latency", 0)
    latency_delta = v2_latency - v1_latency
    
    # Pass rate
    v1_total = max(v1_metrics.get("total_cases", 0), 1)
    v2_total = max(v2_metrics.get("total_cases", 0), 1)
    v1_pass_rate = 100 * v1_metrics.get("pass_count", 0) / v1_total
    v2_pass_rate = 100 * v2_metrics.get("pass_count", 0) / v2_total
    
    print(f"\n📈 Quality:")
    print(f"  V1 Avg Score: {v1_score:.4f}")
    print(f"  V2 Avg Score: {v2_score:.4f}")
    print(f"  Delta: {'+' if score_delta >= 0 else ''}{score_delta:.4f}")
    print(f"  V1 Pass Rate: {v1_pass_rate:.1f}%")
    print(f"  V2 Pass Rate: {v2_pass_rate:.1f}%")
    
    print(f"\n💰 Cost:")
    print(f"  V1 Total Cost: ${v1_cost:.6f}")
    print(f"  V2 Total Cost: ${v2_cost:.6f}")
    print(f"  Cost Delta: ${cost_delta:.6f} ({cost_increase_pct:+.1f}%)")
    print(f"  Cost Ratio (V2/V1): {cost_ratio:.2f}x")
    
    print(f"\n⏱️  Latency:")
    print(f"  V1 Avg Latency: {v1_latency:.4f}s")
    print(f"  V2 Avg Latency: {v2_latency:.4f}s")
    print(f"  Latency Delta: {latency_delta:+.4f}s")
    
    # Regression Gate Logic (P0: Cost threshold, P1: Quality improvement)
    print(f"\n🚪 REGRESSION GATE DECISION:")
    
    decision = "BLOCK"
    reason = ""
    
    # Check 1: Quality improvement or neutral
    if score_delta < -0.05:  # Quality decreased significantly
        reason = f"Quality degraded by {abs(score_delta):.4f}"
        decision = "BLOCK"
    
    # Check 2: Cost threshold (30% is acceptable max)
    elif cost_ratio > 1.30:
        reason = f"Cost increased {cost_increase_pct:.1f}% (threshold: 30%)"
        decision = "BLOCK"
    
    # Check 3: Quality improvement worth the cost
    elif score_delta > 0.05:
        # Quality improved, cost acceptable
        decision = "APPROVE"
        reason = f"Quality improved by {score_delta:.4f}, cost ratio {cost_ratio:.2f}x acceptable"
    
    # Check 4: Cost-neutral or cost-reduced
    elif cost_ratio <= 1.1:  # Cost increased < 10%
        # Cost acceptable or reduced, score neutral/improved
        decision = "APPROVE"
        reason = f"Cost ratio {cost_ratio:.2f}x acceptable, quality neutral/improved"
    
    else:
        decision = "BLOCK"
        reason = f"Cost-quality tradeoff unfavorable: {cost_increase_pct:.1f}% cost increase for {score_delta:+.4f} quality"
    
    print(f"\nDecision: {'✅ APPROVE RELEASE' if decision == 'APPROVE' else '❌ BLOCK RELEASE'}")
    print(f"Reason: {reason}")
    
    regression_metrics = {
        "v1_avg_score": v1_score,
        "v2_avg_score": v2_score,
        "score_delta": score_delta,
        "v1_cost_usd": v1_cost,
        "v2_cost_usd": v2_cost,
        "cost_delta_usd": cost_delta,
        "cost_ratio": round(cost_ratio, 4),
        "cost_increase_pct": cost_increase_pct,
        "v1_latency": v1_latency,
        "v2_latency": v2_latency,
        "latency_delta_seconds": round(latency_delta, 4),
        "v1_pass_rate": v1_pass_rate,
        "v2_pass_rate": v2_pass_rate,
        "decision": decision,
        "reason": reason
    }
    
    return decision, regression_metrics


def print_cost_analysis(v1_results: List[Dict], v2_results: List[Dict]):
    """
    Print cost analysis and propose optimization strategies.
    """
    print("\n💡 --- COST ANALYSIS & OPTIMIZATION PROPOSALS ---")
    
    v1_cases = [r for r in v1_results if r.get("status") == "pass"]
    v2_cases = [r for r in v2_results if r.get("status") == "pass"]
    
    if not v1_cases or not v2_cases:
        print("Insufficient data for cost analysis.")
        return
    
    v1_avg_cost_per_case = sum(r.get("cost_usd", 0) for r in v1_cases) / len(v1_cases)
    v2_avg_cost_per_case = sum(r.get("cost_usd", 0) for r in v2_cases) / len(v2_cases)
    
    v1_avg_tokens = sum(r.get("tokens_used", 0) for r in v1_cases) / len(v1_cases)
    v2_avg_tokens = sum(r.get("tokens_used", 0) for r in v2_cases) / len(v2_cases)
    
    print(f"\n📉 Cost per Case:")
    print(f"  V1 Avg: ${v1_avg_cost_per_case:.6f} ({v1_avg_tokens:.0f} tokens)")
    print(f"  V2 Avg: ${v2_avg_cost_per_case:.6f} ({v2_avg_tokens:.0f} tokens)")
    print(f"  Increase: ${v2_avg_cost_per_case - v1_avg_cost_per_case:.6f}")
    
    print(f"\n💡 Optimization Proposals to Reduce Cost by 30%:")
    print(f"  1. Cache judge responses for identical/similar cases")
    print(f"     → Est. savings: 15-20% (skip redundant judge calls)")
    print(f"  2. Use gpt-4o-mini for retrieval evaluation (not gpt-4o)")
    print(f"     → Est. savings: 5-10% (cheaper model for simpler task)")
    print(f"  3. Batch judge calls in groups of 5 instead of per-case")
    print(f"     → Est. savings: 3-5% (batch API efficiency)")
    print(f"  4. Implement early exit for clearly failed cases (score < 1.5)")
    print(f"     → Est. savings: 5-8% (skip detailed analysis for certain fails)")
    print(f"\n  Total Estimated Savings: 28-43% (target 30%+)")


def save_reports(
    v1_results: List[Dict],
    v2_results: List[Dict],
    regression_metrics: Optional[Dict] = None
):
    """
    Save unified benchmark summary with V1/V2 metrics merged,
    plus detailed benchmark results and regression analysis.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Compute metrics for V1 and V2
    v1_metrics = compute_metrics(v1_results)
    v2_metrics = compute_metrics(v2_results)
    
    dataset_size = len(v1_results) if v1_results else len(v2_results)
    
    # summary.json must remain compatible with check_lab.py:
    # - metadata.total
    # - metrics.avg_score
    # - metrics.hit_rate
    # - metrics.agreement_rate
    # We expose V2 as the candidate release metrics and keep V1/V2 comparison below.
    checklab_summary = {
        "metadata": {
            "version": "regression_v1_vs_v2",
            "total": dataset_size,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "agent_v1": "MainAgent (gpt-4o)",
            "agent_v2": "MainAgentV2 (gpt-4.1)"
        },
        "metrics": {
            "avg_score": v2_metrics.get("avg_score", 0.0),
            "pass_count": v2_metrics.get("pass_count", 0),
            "fail_count": v2_metrics.get("fail_count", 0),
            "pass_rate": v2_metrics.get("pass_rate", 0.0),
            "hit_rate": v2_metrics.get("hit_rate", 0.0),
            "mrr": v2_metrics.get("avg_mrr", 0.0),
            "faithfulness": v2_metrics.get("avg_faithfulness", 0.0),
            "relevancy": v2_metrics.get("avg_relevancy", 0.0),
            "agreement_rate": v2_metrics.get("agreement_rate", 0.0),
            "avg_latency": v2_metrics.get("avg_latency", 0.0),
            "total_cost": v2_metrics.get("total_cost", 0.0),
            "total_tokens": v2_metrics.get("total_tokens", 0)
        },
        "comparison": {
            "v1": v1_metrics,
            "v2": v2_metrics
        }
    }
    
    # Add regression if provided
    if regression_metrics:
        checklab_summary["regression"] = {
            "decision": regression_metrics["decision"],
            "reason": regression_metrics["reason"],
            "v1_avg_score": regression_metrics.get("v1_avg_score", 0),
            "v2_avg_score": regression_metrics.get("v2_avg_score", 0),
            "score_delta": regression_metrics.get("score_delta", 0),
            "cost_ratio": regression_metrics.get("cost_ratio", 0),
            "latency_delta_seconds": regression_metrics.get("latency_delta_seconds", 0)
        }
    
    # Save unified summary
    with open(os.path.join(REPORT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(checklab_summary, f, ensure_ascii=False, indent=2)

    # Save V2 detailed results (full benchmark data)
    with open(os.path.join(REPORT_DIR, "benchmark_results.json"), "w", encoding="utf-8") as f:
        # Include both v1 and v2 results for reference
        benchmark_data = {
            "v1_results": v1_results,
            "v2_results": v2_results
        }
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)


# =========================================================
# MAIN
# =========================================================
async def main():
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        return

    # Run V1
    print("\n" + "="*60)
    print("PHASE 1: Running V1 Baseline...")
    print("="*60)
    v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base", dataset)

    # Run V2
    print("\n" + "="*60)
    print("PHASE 2: Running V2 Optimized...")
    print("="*60)
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized",
        dataset
    )

    if not v1_summary or not v2_summary:
        print("❌ Benchmark failed. Check dataset.")
        return

    # Phase 3: Regression Analysis
    print("\n" + "="*60)
    print("PHASE 3: Regression Analysis & Cost Review...")
    print("="*60)
    decision, regression_metrics = print_comparison(v1_summary, v2_summary)
    
    # Phase 4: Cost Analysis
    print("\n" + "="*60)
    print("PHASE 4: Cost Optimization Analysis...")
    print("="*60)
    print_cost_analysis(v1_results, v2_results)

    # Save all reports
    print("\n" + "="*60)
    print("PHASE 5: Saving Reports...")
    print("="*60)
    save_reports(v1_results, v2_results, regression_metrics)
    print(f"✅ Reports saved to {REPORT_DIR}/")
    print(f"   - summary.json (unified V1/V2 metrics + regression)")
    print(f"   - benchmark_results.json (detailed results)")


if __name__ == "__main__":
    asyncio.run(main())