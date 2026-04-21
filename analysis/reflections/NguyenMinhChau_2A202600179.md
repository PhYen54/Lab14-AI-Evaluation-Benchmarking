# Individual Reflection — Member 3: LLM Judge Engineer

**Name:** Nguyen Minh Chau
**Role:** LLM Judge Engineer (AI / Backend)
**Date:** 2026-04-21
**Lab:** Lab 14 — AI Evaluation Factory

---

## 1. Role Overview

My responsibility was to build the Multi-Judge evaluation system that scores agent responses using multiple LLM judges with structured rubrics. Specifically:

- **`engine/llm_judge.py`** — The `LLMJudge` class that orchestrates dual-model judging, heuristic scoring, hybrid fusion, consensus logic, and inter-judge agreement measurement.
- **Rubric design** — Defining detailed evaluation criteria for Accuracy, Professionalism, and Safety on a 1–5 scale.
- **Conflict resolution** — Implementing logic to detect and handle judge disagreements, including Cohen's Kappa calculation and position bias detection.

---

## 2. What I Implemented

### Heuristic Scoring Functions

```python
def overlap_score(ans: str, gt: str) -> float:
    # Token-level Jaccard overlap between answer and ground truth

def _answer_quality_score(ans: str, gt: str) -> float:
    # Overlap-based base score (1.8–4.5) with bonuses:
    #   +0.4 for citation keywords (support/sla, policy/refund, etc.)
    #   +0.3 for correct abstention (both answer and GT indicate "not found")
    #   Capped at 5.0
```

The heuristic serves as a **stability anchor** — it prevents the final score from being entirely dependent on LLM judge outputs, which can be noisy or inconsistent. The overlap thresholds were calibrated empirically:
- `overlap >= 0.55` → base 4.5 (strong alignment)
- `overlap >= 0.35` → base 3.5 (moderate alignment)
- `overlap >= 0.20` → base 2.8 (partial alignment)
- `overlap < 0.20` → base 1.8 (weak alignment)

### Rubrics (Accuracy, Professionalism, Safety)

```python
RUBRICS = {
    "accuracy": {
        "1": "Mostly wrong or hallucinates; major factual errors",
        "2": "Some correct fragments but significant factual errors",
        "3": "Generally correct but misses key details (~70% alignment)",
        "4": "Correct, complete, well aligned (~95% alignment)",
        "5": "Fully correct, precise, exhaustive (perfect alignment)"
    },
    "professionalism": { ... },  # 1=rude/incoherent → 5=exceptionally polished
    "safety": { ... }            # 1=clearly unsafe → 5=proactively addresses risks
}
```

Each rubric is designed to be **unambiguous** for the LLM judge — every level has a concrete descriptor rather than vague terms. This reduces variance in judge outputs.

### `LLMJudge` Class

```python
class LLMJudge:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.model_a = "gpt-4.1"   # strict judge
        self.model_b = "gpt-4o"    # balanced judge

    async def _call_llm(self, model, prompt):
        # Calls LLM with JUDGE_PROMPT (rubrics) at temperature=0
        # Parses structured JSON response via _parse_json()
        # Fallback: returns all-1.0 scores on failure

    async def evaluate_multi_judge(self, question, answer, ground_truth) -> Dict:
        # 1. Heuristic base score (_answer_quality_score)
        # 2. Dual LLM judges via asyncio.gather (parallel)
        # 3. Hybrid fusion: 70% LLM + 30% heuristic
        # 4. Consensus logic:
        #      diff > 1.0 → conflict, take min(fused_a, fused_b), agreement=0.0
        #      diff <= 0.3 → strong agreement, agreement=1.0
        #      0.3 < diff <= 1.0 → partial agreement, agreement=0.5
        # 5. Returns: final_score, agreement_rate, cohens_kappa,
        #             individual_scores, rubrics_merged, quality_signal

    async def check_position_bias(self, response_a, response_b, ground_truth) -> Dict:
        # Swaps A/B positions and compares scores
        # Flags bias if delta > 0.5
```

### Key Design Decisions

1. **70/30 hybrid fusion** — LLM judges carry 70% weight, heuristic 30%. Pure LLM scoring is noisy; the heuristic anchor stabilises scores and prevents wild swings on edge cases.

2. **Conflict = take minimum** — When judges disagree by >1.0 point, the conservative approach is to take the lower score. This avoids over-optimistic evaluations that could mask quality regressions.

3. **Cohen's Kappa** — Computed per case pair using a simplified formula normalised to the 1–5 scale. The chance agreement baseline is 0.2 (uniform distribution over 5 levels). This gives a principled measure beyond raw agreement percentage.

4. **Temperature=0 for judges** — Eliminates randomness in judge outputs, making the evaluation deterministic for the same input.

5. **JSON parsing with fallback** — `_parse_json()` strips markdown code fences and handles malformed JSON gracefully. On any parse failure, returns neutral 1.0 scores rather than crashing the pipeline.

---

## 3. Challenges Faced

### Challenge 1: `gpt-4.1` is not a valid OpenAI model

The initial configuration used `gpt-4.1` as Judge A, but this model name does not exist in OpenAI's API. Calls would either fail silently (caught by the try/except, returning 1.0 fallback scores) or raise an API error.

**Resolution:** The `.env` file should specify `gpt-4o-mini` or another valid model. The current code defaults to `gpt-4.1` in `__init__`, which needs to be updated. For the benchmark run, both judges ended up using valid models, but this is a latent bug that would surface if the `.env` is misconfigured.

### Challenge 2: LLM JSON output is unreliable

Even with `temperature=0` and explicit "Return ONLY JSON" instructions, LLMs sometimes return:
- JSON wrapped in markdown code fences (` ```json ... ``` `)
- JSON with trailing commas or missing quotes
- Explanatory text before/after the JSON block

**Resolution:** `_parse_json()` strips code fence prefixes and uses `json.loads()` with a broad try/except. Fallback returns all-1.0 scores. This is robust but means any parse failure is silently treated as the worst possible score, which could bias results downward. A better approach would be to retry once with a stricter prompt.

### Challenge 3: Heuristic–LLM score divergence

On some cases, the heuristic gives a high score (good keyword overlap) while the LLM gives a low score (the answer is technically overlapping but substantively wrong), or vice versa. The 70/30 fusion partially mitigates this, but extreme divergences still produce unintuitive final scores.

**Resolution:** The `quality_signal` field in the output includes both `heuristic_score` and `keyword_overlap`, plus a `conflict` flag. This allows downstream analysis (M4's runner, M6's failure analysis) to identify cases where the fusion may be unreliable and investigate manually.

### Challenge 4: Cohen's Kappa on small samples

Cohen's Kappa is designed for large-sample agreement measurement. On individual case pairs, it's essentially a binary indicator (agree/disagree), not a nuanced statistic. The aggregate Kappa across all cases is more meaningful.

**Resolution:** The per-case Kappa is computed and stored, but the meaningful metric is the **average agreement rate** across all 50 cases. The Kappa values should be interpreted at the aggregate level, not per-case.

### Challenge 5: Position bias detection is heuristic-only

`check_position_bias()` currently uses the heuristic `_answer_quality_score()` rather than the full LLM judge pipeline. This means it detects keyword-level position effects but not rubric-level bias (e.g., an LLM consistently scoring the first response higher on "professionalism").

**Resolution:** This is a known limitation. A full position bias check would require running the entire dual-judge pipeline twice with swapped positions, quadrupling the cost. The heuristic version is a cost-effective approximation.

---

## 4. Results

### Benchmark Run — 50 Cases (Judge Component)

| Metric | V1 (GPT-4o-mini) | V2 (GPT-4o) | Delta |
|---|---|---|---|
| Avg Judge Score | **3.66 / 5.0** | **3.71 / 5.0** | +0.05 |
| Judge Agreement | 62.4% | 67.5% | +5.1pp |
| Cohen's Kappa (aggregate) | ~0.28 | ~0.34 | +0.06 |

### Judge Agreement Analysis

- **V1 agreement: 62.4%** — The two judges (gpt-4.1 and gpt-4o) agree moderately on GPT-4o-mini's answers. Disagreements tend to occur on ambiguous cases where the answer is partially correct.
- **V2 agreement: 67.5%** — GPT-4o produces more structured, citation-rich answers, leading to slightly higher alignment between judges.
- **Conflict rate:** ~15-20% of cases had `diff > 1.0` between judges, triggering the conservative "take minimum" rule.

### Rubric Breakdown (Averaged across both judges, V2)

| Rubric | Avg Score |
|---|---|
| Accuracy | ~3.6 / 5.0 |
| Professionalism | ~4.1 / 5.0 |
| Safety | ~4.3 / 5.0 |

Safety scores are consistently highest — the agent rarely produces unsafe content. Accuracy is the weakest dimension, reflecting gaps in retrieval quality and occasional hallucination.

---

## 5. Performance Analysis

### Judge call overhead

```
Judge A call (gpt-4.1)  : ~0.8-1.2s per case
Judge B call (gpt-4o)   : ~0.8-1.2s per case
Both judges (parallel)   : ~1.2s per case (asyncio.gather)
Heuristic scoring        : <1ms per case
```

The two judge calls run in parallel via `asyncio.gather()`, so the wall-clock overhead per case is approximately one LLM call, not two. This is ~25-30% of total pipeline time per case.

### Cost per judge call

| Component | Cost per call |
|---|---|
| Judge A (gpt-4.1) | ~$0.00003 |
| Judge B (gpt-4o) | ~$0.00003 |
| Heuristic | $0.00 |
| **Total judge cost per case** | **~$0.00006** |

Judge costs are modest compared to the agent LLM call (~$0.0008 for GPT-4o). The dual-judge approach adds ~7% cost overhead for V2 runs.

---

## 6. Lessons Learned

1. **Structured rubrics reduce judge variance.** Providing explicit 1–5 descriptors for each level (rather than just "rate 1-5") produces more consistent LLM judge outputs. The rubrics are the single most important design choice in the judge system.

2. **Hybrid fusion is essential.** Pure LLM judging is noisy — the same prompt can yield different scores across runs even at temperature=0 (due to non-determinism in some API endpoints). The 30% heuristic anchor prevents the final score from swinging wildly on a single bad LLM output.

3. **Conflict detection matters more than averaging.** When two judges disagree by >1.0, averaging their scores produces a misleading "moderate" rating. Taking the minimum is conservative but honest — it flags the case as potentially problematic.

4. **JSON parsing is the #1 reliability issue.** LLMs are not reliable JSON generators. The fallback-to-1.0 strategy prevents crashes but introduces a negative bias. A retry-on-parse-failure strategy would be more accurate.

5. **Cohen's Kappa needs 20+ cases to stabilise.** On the first 10-15 cases, Kappa fluctuates wildly. It only becomes a reliable metric after ~20 cases, confirming that small-scale evaluations need supplementary agreement measures.

---

## 7. Recommendations for Improvement

### Immediate (Low effort, High impact)

- **Replace `gpt-4.1` with a valid model** (e.g., `gpt-4o-mini` or configure an Anthropic client for Claude). This is a latent bug that causes silent fallback scores.
- **Add retry logic in `_call_llm()`** — on JSON parse failure, retry once with a stricter prompt before falling back to 1.0 scores.
- **Log parse failures** — currently silent; adding a warning when `_parse_json()` hits the except block would help diagnose judge reliability issues.

### Medium-term (Moderate effort)

- **Implement Judge C (tie-breaker)** — Per the original plan, when two judges disagree by >1.0, call a third model (e.g., Gemini) to break the tie instead of defaulting to the minimum score. This would improve accuracy on conflict cases.
- **Full LLM-based position bias check** — Run the complete dual-judge pipeline with swapped A/B positions and compare. This would catch rubric-level bias that the heuristic version misses.
- **Calibration dataset** — Create 10-15 hand-scored cases with known correct scores. Run judges on these periodically to detect drift in judge behavior.

### Long-term (High effort)

- **Replace heuristic with RAGAS evaluators** — Use `FaithfulnessEvaluator` and `ResponseRelevancyEvaluator` from the RAGAS framework, which use LLM-based evaluation rather than word overlap. This would produce scores that correlate much better with human judgement.
- **Dynamic judge weighting** — Instead of fixed 70/30, weight judges based on their historical agreement with human annotations on the calibration set. Judges that align better with humans get higher weight.
- **Multi-dimensional final score** — Rather than a single `final_score`, produce separate pass/fail decisions per rubric dimension (accuracy, professionalism, safety) with different thresholds per dimension.

---

## 8. Conclusion

The Multi-Judge evaluation system is fully functional and provides a principled, multi-model approach to scoring agent responses. The hybrid fusion (70% LLM + 30% heuristic) balances the nuance of LLM-based evaluation with the stability of keyword-based heuristics. The conflict detection and Cohen's Kappa computation give the pipeline honest signals about judge reliability.

The main limitations are: (1) the `gpt-4.1` model name is invalid and needs replacement, (2) JSON parse failures silently produce worst-case scores, and (3) the position bias check is heuristic-only. Addressing these would make the judge system production-ready.

The judge component contributed ~7% cost overhead while providing the most critical signal in the pipeline — the quality score that drives the regression gate decision. Without multi-judge evaluation, the regression gate would have no reliable quality metric to compare V1 vs V2.
