import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

STUDENT_CSV      = "cleaned_dataset.csv"
EXPECTED_CSV     = "expected_answers_dual.csv"
OUTPUT_RAW       = "batch_results_v3.csv"
OUTPUT_CALIBRATED= "batch_results_calibrated.csv"
OUTPUT_FLAGGED   = "flagged_for_review.csv"

LLM_RETRY_ATTEMPTS   = 3      # retry on JSON parse failure
SLEEP_BETWEEN_CALLS  = 1.0    # seconds between API calls
MAX_STUDENT_CHARS    = 3000   # truncate very long answers to avoid silent LLM drops


# =============================================================================
# 2. EXPECTED ANSWER SELECTOR
# =============================================================================

def select_expected_answer(expected_essay: str, expected_visual: str) -> tuple[str, str]:
    """
    Pick expected_essay if available, otherwise expected_visual.
    Simple rule — no embeddings needed.
    """
    essay_ok  = expected_essay  and expected_essay.strip()  not in ("", "nan")
    visual_ok = expected_visual and expected_visual.strip() not in ("", "nan")

    if essay_ok:
        return expected_essay, "essay"
    elif visual_ok:
        return expected_visual, "visual"
    return "", "none"


# =============================================================================
# 4. LLM SCORER  (retry logic + confidence + structured concept extraction)
# =============================================================================

def get_llm_score(
    student_ans: str,
    expected_ans: str,
    max_marks: float,
    question: str,
    rubric: str
) -> tuple[float, str, list]:
    """
    Returns (score, confidence, concepts_list).

    confidence: "high" | "medium" | "low"
    concepts_list: list of dicts [{"concept": ..., "status": "full|partial|missing"}]
    """

    # Truncate very long answers to prevent silent concept-extraction failure
    if len(student_ans) > MAX_STUDENT_CHARS:
        student_ans = student_ans[:MAX_STUDENT_CHARS] + "\n...[answer truncated for length]"

    # Scale concept target to question size so big questions aren't over-penalised
    min_concepts = max(3, int(max_marks / 8))
    max_concepts = max(6, int(max_marks / 4))

    prompt = f"""
You are a fair and experienced university examiner grading a {max_marks}-mark question.

IMPORTANT CONTEXT:
The REFERENCE ANSWER below was written at an advanced academic level — it is a MODEL answer,
not a student answer. Students are NOT expected to use the same terminology, phrasing, or
level of detail. Your job is to assess whether the STUDENT UNDERSTANDS the underlying concept,
regardless of how simply or informally they express it.

If a student explains a concept correctly in plain, everyday language, that is FULL CREDIT.
Penalise only for factual incorrectness or genuine conceptual gaps — never for informal wording.

QUESTION:
{question}

REFERENCE ANSWER (advanced model answer — use only as a concept map, not as a wording benchmark):
{expected_ans}

STUDENT ANSWER:
{student_ans}

RUBRIC:
{rubric if rubric and rubric.strip() not in ("", "nan") else "Conceptual understanding and correctness."}

=== GRADING INSTRUCTIONS ===

STEP 1 — BUILD A CONCEPT MAP from the REFERENCE ANSWER.
  - Distil it into core ideas that answer the QUESTION.
  - Strip away advanced vocabulary — ask: "what is the underlying idea here?"
  - Target between {min_concepts} and {max_concepts} concepts for a {max_marks}-mark question.
  - Each concept should be expressible in one plain sentence.

STEP 2 — CHECK STUDENT UNDERSTANDING for each concept.
  Ask yourself: "Does the student show they understand this idea,
  even if they say it differently or more simply?"

  - "full"    → student demonstrates clear understanding of the concept,
                even if using simple, informal, or different words         → 1.0 credit
  - "partial" → student shows some awareness but is incomplete,
                vague, or slightly off                                     → 0.5 credit
  - "missing" → student shows no understanding of this concept,
                or states something factually incorrect                    → 0.0 credit

STEP 3 — COMPUTE SCORE:
  credited = sum of credits across all concepts
  score    = (credited / total_concepts) * {max_marks}
  Round to nearest 0.25.

CRITICAL RULES:
  - NEVER penalise a student for not using technical or advanced terminology.
  - NEVER penalise for brevity if the core understanding is present.
  - A simple but correct explanation = full credit for that concept.
  - Only deduct for factual errors or missing conceptual understanding.
  - A completely blank, illegible, or entirely off-topic answer scores 0.
  - Do NOT reward padding or repetition that adds no new understanding.

STEP 4 — ASSIGN CONFIDENCE:
  - "high"   → concepts were clear and matching was straightforward
  - "medium" → some ambiguity in what the student meant, or borderline cases
  - "low"    → reference answer is very vague, student answer is unreadable,
               or you genuinely cannot tell if the student understands

Return ONLY valid JSON (no markdown, no extra text):
{{
  "concepts": [
    {{
      "concept": "<plain-language description of the underlying idea>",
      "status": "full|partial|missing",
      "student_evidence": "<brief quote or paraphrase of what the student said, or 'nothing'"
    }}
  ],
  "total_concepts": <integer>,
  "credited_concepts": <float>,
  "score": <float>,
  "confidence": "high|medium|low"
}}
"""

    for attempt in range(1, LLM_RETRY_ATTEMPTS + 1):
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            clean = response.text.strip().replace("```json", "").replace("```", "").strip()
            data  = json.loads(clean)

            score      = float(data["score"])
            confidence = str(data.get("confidence", "medium")).lower()
            concepts   = data.get("concepts", [])

            # Sanity clamp
            score = min(max(score, 0.0), max_marks)

            return score, confidence, concepts

        except json.JSONDecodeError as e:
            print(f"  [Attempt {attempt} JSON error: {e}]", end="")
        except Exception as e:
            print(f"  [Attempt {attempt} error: {e}]", end="")

        if attempt < LLM_RETRY_ATTEMPTS:
            time.sleep(2.0 * attempt)   # exponential back-off

    # All retries exhausted → flag as low confidence, return 0
    print("  [ALL RETRIES FAILED — scored 0, flagged low confidence]", end="")
    return 0.0, "low", []


# =============================================================================
# 5. LOAD AND MERGE DATA
# =============================================================================

print("📂 Loading datasets...")
df_students = pd.read_csv(STUDENT_CSV)
df_expected = pd.read_csv(EXPECTED_CSV)

# Drop duplicate columns before merge
common_cols = [
    c for c in df_students.columns
    if c in df_expected.columns and c != "question_id"
]
df_students = df_students.drop(columns=common_cols)
df_merged = pd.merge(df_students, df_expected, on="question_id", how="left")

# Normalise column names from merge suffixes
for col in ["marking_scheme", "max_marks"]:
    if f"{col}_y" in df_merged.columns:
        df_merged.rename(columns={f"{col}_y": col}, inplace=True)
    if f"{col}_x" in df_merged.columns and col not in df_merged.columns:
        df_merged.rename(columns={f"{col}_x": col}, inplace=True)

print(f"✅ Loaded {len(df_merged)} answer scripts across "
      f"{df_merged['question_id'].nunique()} questions.\n")


# =============================================================================
# 6. MAIN GRADING LOOP
# =============================================================================

print(f"{'='*60}")
print(f"  🤖 Hybrid AES Grading — {len(df_merged)} scripts")
print(f"{'='*60}\n")

final_results       = []
low_confidence_flags = []

for index, row in df_merged.iterrows():
    qid          = str(row["question_id"])
    student_text = str(row.get("extracted_text", ""))
    q_text       = str(row.get("question_text", ""))
    rubric       = str(row.get("marking_scheme", ""))

    try:
        max_marks = float(row["max_marks"])
    except (ValueError, KeyError):
        max_marks = 10.0

    print(f"  [{index+1:>3}/{len(df_merged)}] {qid} (/{max_marks})...", end="", flush=True)

    expected_essay  = str(row.get("expected_essay",  ""))
    expected_visual = str(row.get("expected_visual", ""))

    # ── Select expected answer (essay preferred, visual as fallback) ───────
    expected_answer, answer_type = select_expected_answer(expected_essay, expected_visual)

    if not expected_answer:
        print(" [No valid expected answer — skipping]", end="")
        final_score = 0.0
        confidence  = "low"
        concepts    = []
    else:
        # ── LLM concept scoring ────────────────────────────────────────────
        llm_score, confidence, concepts = get_llm_score(
            student_text,
            expected_answer,
            max_marks,
            q_text,
            rubric
        )

        # Round to nearest 0.25 mark (standard academic practice)
        final_score = round(round(llm_score * 4) / 4, 2)
        final_score = min(max(final_score, 0.0), max_marks)

    print(f"  score={final_score}  conf={confidence}  ref={answer_type}  concepts={len(concepts)}")

    result_row = {
        "question_id"  : qid,
        "human_score"  : row.get("teacher_score", np.nan),
        "ai_score"     : final_score,
        "max_marks"    : max_marks,
        "answer_type"  : answer_type,
        "num_concepts" : len(concepts),
        "confidence"   : confidence,
        "method"       : "LLM Concept v3"
    }

    final_results.append(result_row)

    if confidence == "low":
        low_confidence_flags.append({**result_row, "concepts_detail": json.dumps(concepts)})

    time.sleep(SLEEP_BETWEEN_CALLS)

# ── Save raw results ────────────────────────────────────────────────────────
df_results = pd.DataFrame(final_results)
df_results.to_csv(OUTPUT_RAW, index=False)
print(f"\n✅ Raw results saved to '{OUTPUT_RAW}'")
print(f"   {len(low_confidence_flags)} scripts flagged as low confidence.")


# =============================================================================
# 7. PER-QUESTION CALIBRATION  (corrects systematic underscoring bias)
# =============================================================================

print("\n📐 Running per-question calibration...")

df_cal = df_results.copy()
calibration_log = {}

calibrated_rows = []

for qid, group in df_cal.groupby("question_id"):
    graded = group.dropna(subset=["human_score"])

    if len(graded) < 3:
        # Not enough data to calibrate — keep raw scores as-is
        group["calibrated_score"] = group["ai_score"]
        group["cal_scale"]        = 1.0
        group["cal_shift"]        = 0.0
        calibration_log[qid]      = {"scale": 1.0, "shift": 0.0, "note": "insufficient_data"}
    else:
        ai_vals = graded["ai_score"].astype(float).values
        hu_vals = graded["human_score"].astype(float).values

        # Guard: if all ai_scores are identical (e.g. all 0 due to embedding failure)
        # polyfit would produce a degenerate matrix → SVD crash.
        # Fall back to a shift-only correction in that case.
        if np.std(ai_vals) < 1e-6:
            shift = float(np.mean(hu_vals) - np.mean(ai_vals))
            scale = 1.0
            note  = "degenerate_input_shift_only"
            print(f"   {qid:8s}  ⚠ all AI scores identical — shift-only calibration applied")
        else:
            try:
                coeffs = np.polyfit(ai_vals, hu_vals, 1)
                scale, shift = float(coeffs[0]), float(coeffs[1])
                note = "ok"
            except np.linalg.LinAlgError:
                scale, shift = 1.0, float(np.mean(hu_vals) - np.mean(ai_vals))
                note = "svd_fallback"
                print(f"   {qid:8s}  ⚠ polyfit SVD failed — shift-only fallback used")

        # Clamp to prevent wild extrapolation from a small calibration set
        scale = float(np.clip(scale, 0.75, 1.75))
        shift = float(np.clip(shift, -10.0, 10.0))

        max_m = float(group["max_marks"].iloc[0])
        group["calibrated_score"] = (
            (group["ai_score"] * scale + shift)
            .clip(0, max_m)
            .round(2)
        )
        group["cal_scale"] = scale
        group["cal_shift"] = shift
        calibration_log[qid] = {"scale": round(scale, 4), "shift": round(shift, 4), "n": len(graded), "note": note}

        if note == "ok":
            print(f"   {qid:8s}  scale={scale:.3f}  shift={shift:+.3f}  (n={len(graded)})")

    calibrated_rows.append(group)

df_calibrated = pd.concat(calibrated_rows).reset_index(drop=True)
df_calibrated.to_csv(OUTPUT_CALIBRATED, index=False)
print(f"\n✅ Calibrated results saved to '{OUTPUT_CALIBRATED}'")


# =============================================================================
# 8. SAVE FLAGGED SCRIPTS FOR HUMAN REVIEW
# =============================================================================

if low_confidence_flags:
    df_flagged = pd.DataFrame(low_confidence_flags)
    df_flagged.to_csv(OUTPUT_FLAGGED, index=False)
    print(f"⚠️  Flagged scripts saved to '{OUTPUT_FLAGGED}'")
else:
    print("✅ No low-confidence scripts to flag.")


# =============================================================================
# 9. EVALUATION METRICS
# =============================================================================

def print_metrics(df: pd.DataFrame, score_col: str, label: str):
    eval_df = df.dropna(subset=["human_score", score_col])
    if len(eval_df) == 0:
        print(f"  No scored rows available for {label}.")
        return

    h = eval_df["human_score"].astype(float)
    a = eval_df[score_col].astype(float)

    mae  = mean_absolute_error(h, a)
    rmse = float(np.sqrt(np.mean((h - a) ** 2)))
    r, _ = pearsonr(h, a)

    within_1_mark  = (np.abs(h - a) <= 1).mean() * 100
    within_5_marks = (np.abs(h - a) <= 5).mean() * 100

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  N scripts          : {len(eval_df)}")
    print(f"  MAE                : {mae:.3f} marks")
    print(f"  RMSE               : {rmse:.3f}")
    print(f"  Pearson r          : {r:.3f}")
    print(f"  Within ±1 mark     : {within_1_mark:.1f}%")
    print(f"  Within ±5 marks    : {within_5_marks:.1f}%")

    # Top 5 worst errors
    eval_df = eval_df.copy()
    eval_df["abs_diff"] = np.abs(h - a)
    top5 = eval_df.nlargest(5, "abs_diff")[
        ["question_id", "human_score", score_col, "abs_diff"]
    ]
    print(f"\n  TOP 5 WORST ERRORS:")
    print(f"  {'Q_ID':<10} {'Human':>6} {'AI':>6} {'Diff':>6}")
    print(f"  {'─'*32}")
    for _, r2 in top5.iterrows():
        print(f"  {r2['question_id']:<10} {r2['human_score']:>6.1f} "
              f"{r2[score_col]:>6.1f} {r2['abs_diff']:>6.2f}")


print(f"\n{'='*50}")
print("  📊 EVALUATION RESULTS")
print(f"{'='*50}")

print_metrics(df_results,     "ai_score",          "RAW AI Scores")
print_metrics(df_calibrated,  "calibrated_score",  "CALIBRATED Scores")

print(f"\n{'='*50}")
print("  🏁 PIPELINE COMPLETE")
print(f"{'='*50}")
print(f"  Raw scores     → {OUTPUT_RAW}")
print(f"  Calibrated     → {OUTPUT_CALIBRATED}")
print(f"  Flagged        → {OUTPUT_FLAGGED}")
print(f"  Calibration log:")
for qid, info in calibration_log.items():
    print(f"    {qid}: {info}")