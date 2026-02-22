import pandas as pd
import numpy as np
import time
import json
import re
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

STUDENT_CSV       = "cleaned_dataset.csv"
EXPECTED_CSV      = "expected_answers_dual.csv"
OUTPUT_RAW        = "batch_results_v4.csv"
OUTPUT_CALIBRATED = "batch_results_v4_calibrated.csv"
OUTPUT_FLAGGED    = "flagged_for_review_v4.csv"

LLM_RETRY_ATTEMPTS  = 3
SLEEP_BETWEEN_CALLS = 1.0
MAX_STUDENT_CHARS   = 3000


# =============================================================================
# 2. EXPECTED ANSWER SELECTOR — UNCHANGED FROM V3
# =============================================================================

def select_expected_answer(expected_essay: str, expected_visual: str) -> tuple[str, str]:
    """
    Pick expected_essay if available, otherwise expected_visual.
    Simple rule — no embeddings needed.
    UNCHANGED from v3.
    """
    essay_ok  = expected_essay  and expected_essay.strip()  not in ("", "nan")
    visual_ok = expected_visual and expected_visual.strip() not in ("", "nan")

    if essay_ok:
        return expected_essay, "essay"
    elif visual_ok:
        return expected_visual, "visual"
    return "", "none"


# =============================================================================
# FIX 5 — OCR DIAGRAM CLEANER
# Strips [[Diagram: ...]] OCR artifacts from student text before grading.
# Replaces them with a short marker so the LLM knows a diagram was present.
# =============================================================================

def clean_ocr_text(text: str) -> str:
    """
    Remove diagram description artifacts produced by OCR.
    Replaces [[Diagram: long description...]] with [student diagram]
    so the grader knows a diagram exists but is not confused by
    hundreds of characters of structural description.
    """
    # Pattern: [[anything]] including multiline
    cleaned = re.sub(r'\[\[.*?\]\]', '[student diagram]', text, flags=re.DOTALL)

    # Also catch (very long parenthetical descriptions > 150 chars)
    # These are often OCR artifacts from figure captions
    cleaned = re.sub(
        r'\(([^)]{150,})\)',
        '(see diagram)',
        cleaned
    )

    return cleaned.strip()


# =============================================================================
# FIX 2 — RELEVANCE PRE-CHECK
# Runs before main LLM scoring. Returns (score, confidence, flag_reason)
# if the answer should be short-circuited, or None to proceed normally.
# =============================================================================

def precheck_answer(student_text: str, question_text: str, max_marks: float):
    """
    Gate 1: Catch blank or near-blank answers immediately.
    Gate 2: Ask LLM a fast YES/NO relevance question.

    Returns (0.0, "low", flag_reason) if answer should score 0,
    or None to proceed to full grading.
    """

    # Gate 1 — too short to be a real answer
    # Strip diagram markers and whitespace before length check
    stripped = re.sub(r'\[student diagram\]', '', student_text).strip()
    if len(stripped) < 30:
        print(" [PRE-CHECK: blank/too short → 0]", end="")
        return 0.0, "low", "flagged_blank"

    # Gate 2 — fast relevance check via LLM
    # Only trigger for higher-mark questions (not worth the API call for 10-mark Qs)
    if max_marks >= 20:
        try:
            relevance_prompt = f"""You are checking if a student answer is relevant to a question.

Question: {question_text}

Student Answer (first 400 characters): {student_text[:400]}

Is this student answer making a genuine attempt to address the question?
Reply with ONLY one word: YES or NO.
- YES: answer attempts to address the question, even if partially or incorrectly
- NO: answer is completely blank, gibberish, entirely off-topic, or unrelated subject matter"""

            model = genai.GenerativeModel("gemini-2.5-flash")
            resp  = model.generate_content(
                relevance_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            answer = resp.text.strip().upper()

            if answer == "NO":
                print(" [PRE-CHECK: irrelevant → 0]", end="")
                return 0.0, "low", "flagged_irrelevant"

        except Exception as e:
            # If relevance check fails, proceed normally — don't block grading
            print(f" [PRE-CHECK error: {e} → proceeding]", end="")

    return None  # proceed to full grading


# =============================================================================
# FIX 1 — SCALED CONCEPT COUNTS (tighter for high-mark questions)
# =============================================================================

def get_concept_bounds(max_marks: float) -> tuple[int, int]:
    """
    Returns (min_concepts, max_concepts) scaled to question mark allocation.

    High-mark essay questions (45+): tight bounds, focus on core ideas only.
    Medium questions (20-44): moderate bounds.
    Short questions (<20): small concept counts.

    This prevents concept inflation where a 55-mark question generates
    10+ concepts and students who cover main ideas get severely underscored.
    """
    if max_marks >= 45:
        # e.g. Q1_c (55 marks): 4-6 concepts max
        min_c = max(4, int(max_marks / 12))
        max_c = max(6, int(max_marks / 8))
    elif max_marks >= 20:
        # e.g. Q4_a (24 marks), Q2_b (45 marks): moderate
        min_c = max(3, int(max_marks / 10))
        max_c = max(5, int(max_marks / 6))
    else:
        # e.g. Q1_a (10 marks), Q2_a (10 marks): small
        min_c = max(2, int(max_marks / 6))
        max_c = max(4, int(max_marks / 4))

    return min_c, max_c


# =============================================================================
# 4. LLM SCORER — v4 with all prompt fixes applied
# FIX 1: Tighter concept scaling via get_concept_bounds()
# FIX 3: Structural knowledge rule (list-only answers)
# FIX 4: Depth verification + domain error + duplication rules
# FIX 5: Diagram-aware grading instructions
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
    concepts_list: list of dicts
    """

    # Truncate very long answers
    if len(student_ans) > MAX_STUDENT_CHARS:
        student_ans = student_ans[:MAX_STUDENT_CHARS] + "\n...[answer truncated for length]"

    # FIX 1: Use tighter concept bounds
    min_concepts, max_concepts = get_concept_bounds(max_marks)

    # Detect if student answer contains diagrams
    has_diagram = '[student diagram]' in student_ans

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
{"  - IMPORTANT: This is a HIGH-MARK essay question. Focus ONLY on the " + str(max_concepts) + " most central, essential concepts." if max_marks >= 45 else ""}
{"  - Ignore peripheral details, advanced elaborations, and optional examples in the reference." if max_marks >= 45 else ""}
{"  - A student who covers the core ideas deserves significant credit even without every detail." if max_marks >= 45 else ""}

STEP 2 — CHECK STUDENT UNDERSTANDING for each concept.

  CREDIT LEVELS:
  - "full"    (1.0 credit): Student demonstrates clear understanding of the concept,
                            even if using simple, informal, or different words.
                            A simple but correct explanation = FULL credit.

  - "partial" (0.5 credit): Student shows some awareness but is incomplete,
                            vague, or slightly imprecise. Also use partial if the
                            student only NAMES the concept without explaining it.

  - "missing" (0.0 credit): Student shows no understanding, states something
                            factually incorrect, or does not address the concept.

  DEPTH VERIFICATION RULE (FIX for annotation inflation):
  Before awarding "full" credit, ask: "Does the student show WHY or HOW this
  concept works — not just restate its name in different words?"
  - Names concept only, no explanation → "partial" at most (0.5)
  - Names + thin/wrong explanation → "partial" (0.5)
  - Names + correct explanation → "full" (1.0)

  STRUCTURAL KNOWLEDGE RULE (FIX for list-only answers):
  If a student correctly lists, names, or enumerates components/steps asked
  for in the question — even without explaining each one — award "partial"
  (0.5) credit as a minimum for each correctly identified item.
  Do NOT give "missing" (0.0) for a correctly named but unexplained concept.
  Correct naming without explanation = "partial", not "missing".

  DOMAIN ERROR RULE (FIX for annotation inflation):
  If a student's example is from a clearly wrong domain
  (e.g., describing a beverage company when the question is about a
  supermarket POS system), downgrade that concept from "full" to "partial".
  Domain errors show partial understanding only.

  DUPLICATION RULE (FIX for annotation inflation):
  If the same step, concept, or idea appears more than once in the student
  answer, count it ONCE only. Do not award credit twice for the same point.

{"  DIAGRAM RULE (this answer contains student diagrams):" if has_diagram else ""}
{"  The marker [student diagram] in the student answer indicates a hand-drawn" if has_diagram else ""}
{"  diagram or sketch. Apply these rules:" if has_diagram else ""}
{"  - If the surrounding written text explains what the diagram shows, award" if has_diagram else ""}
{"    full concept credit based on that context." if has_diagram else ""}
{"  - If the student wrote a definition and references a diagram without further" if has_diagram else ""}
{"    text, award partial credit (0.5) — diagram shows intent but cannot be verified." if has_diagram else ""}
{"  - NEVER give 0 marks solely because an answer is diagram-heavy." if has_diagram else ""}
{"  - Diagram usage is a valid academic choice in handwritten examinations." if has_diagram else ""}

STEP 3 — COMPUTE SCORE:
  credited = sum of credits across all concepts
  score    = (credited / total_concepts) * {max_marks}
  Round to nearest 0.25.

CRITICAL RULES:
  - NEVER penalise a student for not using technical or advanced terminology.
  - NEVER penalise for brevity if the core understanding is present.
  - Only deduct for factual errors or missing conceptual understanding.
  - A completely blank, illegible, or entirely off-topic answer scores 0.
  - Do NOT reward padding or repetition that adds no new understanding.

STEP 4 — ASSIGN CONFIDENCE:
  - "high"   → concepts were clear and matching was straightforward
  - "medium" → some ambiguity in what the student meant, or borderline cases
  - "low"    → reference answer is very vague, student answer is unreadable,
               contains mainly diagrams, or you genuinely cannot tell if
               the student understands

Return ONLY valid JSON (no markdown, no extra text):
{{
  "concepts": [
    {{
      "concept": "<plain-language description of the underlying idea>",
      "status": "full|partial|missing",
      "student_evidence": "<brief quote or paraphrase of what the student said, or 'nothing'>"
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
            model    = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            clean      = response.text.strip().replace("```json", "").replace("```", "").strip()
            data       = json.loads(clean)
            score      = float(data["score"])
            confidence = str(data.get("confidence", "medium")).lower()
            concepts   = data.get("concepts", [])

            score = min(max(score, 0.0), max_marks)
            return score, confidence, concepts

        except json.JSONDecodeError as e:
            print(f"  [Attempt {attempt} JSON error: {e}]", end="")
        except Exception as e:
            print(f"  [Attempt {attempt} error: {e}]", end="")

        if attempt < LLM_RETRY_ATTEMPTS:
            time.sleep(2.0 * attempt)

    print("  [ALL RETRIES FAILED — scored 0, flagged low confidence]", end="")
    return 0.0, "low", []


# =============================================================================
# 5. LOAD AND MERGE DATA
# =============================================================================

print("📂 Loading datasets...")
df_students = pd.read_csv(STUDENT_CSV)
df_expected = pd.read_csv(EXPECTED_CSV)

common_cols = [
    c for c in df_students.columns
    if c in df_expected.columns and c != "question_id"
]
df_students = df_students.drop(columns=common_cols)
df_merged   = pd.merge(df_students, df_expected, on="question_id", how="left")

for col in ["marking_scheme", "max_marks"]:
    if f"{col}_y" in df_merged.columns:
        df_merged.rename(columns={f"{col}_y": col}, inplace=True)
    if f"{col}_x" in df_merged.columns and col not in df_merged.columns:
        df_merged.rename(columns={f"{col}_x": col}, inplace=True)

print(f"✅ Loaded {len(df_merged)} answer scripts across "
      f"{df_merged['question_id'].nunique()} questions.\n")


# =============================================================================
# 6. MAIN GRADING LOOP — v4
# =============================================================================

print(f"{'='*60}")
print(f"  🤖 AES Grading v4 — {len(df_merged)} scripts")
print(f"  Fixes applied:")
print(f"    Fix 1: Tighter concept scaling for high-mark questions")
print(f"    Fix 2: Relevance pre-check for blank/off-topic answers")
print(f"    Fix 3: Structural knowledge rule (list-only answers)")
print(f"    Fix 4: Depth + domain + duplication rules")
print(f"    Fix 5: OCR diagram cleaning + diagram-aware grading")
print(f"{'='*60}\n")

final_results        = []
low_confidence_flags = []
precheck_overrides   = []  # track which scripts were short-circuited by pre-check

for index, row in df_merged.iterrows():
    qid          = str(row["question_id"])
    student_text = str(row.get("extracted_text", ""))
    q_text       = str(row.get("question_text",  ""))
    rubric       = str(row.get("marking_scheme",  ""))

    try:
        max_marks = float(row["max_marks"])
    except (ValueError, KeyError):
        max_marks = 10.0

    print(f"  [{index+1:>3}/{len(df_merged)}] {qid} (/{max_marks})...", end="", flush=True)

    expected_essay  = str(row.get("expected_essay",  ""))
    expected_visual = str(row.get("expected_visual", ""))

    # ── FIX 5: Clean OCR diagram artifacts BEFORE any processing ──────────
    student_text_clean = clean_ocr_text(student_text)

    # ── Select expected answer — UNCHANGED from v3 ─────────────────────────
    expected_answer, answer_type = select_expected_answer(expected_essay, expected_visual)

    if not expected_answer:
        print(" [No valid expected answer — skipping]", end="")
        final_score = 0.0
        confidence  = "low"
        concepts    = []
        flag_reason = "no_expected_answer"

    else:
        # ── FIX 2: Run relevance pre-check BEFORE main LLM call ───────────
        precheck_result = precheck_answer(student_text_clean, q_text, max_marks)

        if precheck_result is not None:
            # Pre-check returned a score — short-circuit grading
            final_score, confidence, flag_reason = precheck_result
            concepts = []
            precheck_overrides.append({
                "question_id": qid,
                "human_score": row.get("teacher_score", np.nan),
                "ai_score"   : final_score,
                "max_marks"  : max_marks,
                "reason"     : flag_reason
            })
        else:
            # ── Normal grading path ────────────────────────────────────────
            flag_reason = None

            llm_score, confidence, concepts = get_llm_score(
                student_text_clean,   # FIX 5: pass cleaned text
                expected_answer,
                max_marks,
                q_text,
                rubric
            )

            final_score = round(round(llm_score * 4) / 4, 2)
            final_score = min(max(final_score, 0.0), max_marks)

    print(f"  score={final_score}  conf={confidence}  ref={answer_type}  "
          f"concepts={len(concepts)}"
          f"{'  [DIAGRAM]' if '[student diagram]' in student_text_clean else ''}")

    result_row = {
        "question_id"  : qid,
        "human_score"  : row.get("teacher_score", np.nan),
        "ai_score"     : final_score,
        "max_marks"    : max_marks,
        "answer_type"  : answer_type,
        "num_concepts" : len(concepts),
        "confidence"   : confidence,
        "has_diagram"  : '[student diagram]' in student_text_clean,
        "method"       : "LLM Concept v4"
    }

    final_results.append(result_row)

    if confidence == "low":
        low_confidence_flags.append({
            **result_row,
            "concepts_detail": json.dumps(concepts),
            "flag_reason"    : flag_reason or "low_confidence"
        })

    time.sleep(SLEEP_BETWEEN_CALLS)

# ── Save raw results ────────────────────────────────────────────────────────
df_results = pd.DataFrame(final_results)
df_results.to_csv(OUTPUT_RAW, index=False)

print(f"\n✅ Raw results saved to '{OUTPUT_RAW}'")
print(f"   {len(low_confidence_flags)} scripts flagged as low confidence.")
print(f"   {len(precheck_overrides)} scripts short-circuited by relevance pre-check.")

if precheck_overrides:
    print("\n  Pre-check overrides:")
    for p in precheck_overrides:
        print(f"    {p['question_id']}  human={p['human_score']}  reason={p['reason']}")


# =============================================================================
# 7. PER-QUESTION CALIBRATION — UNCHANGED from v3
# =============================================================================

print("\n📐 Running per-question calibration...")

df_cal         = df_results.copy()
calibration_log = {}
calibrated_rows = []

for qid, group in df_cal.groupby("question_id"):
    graded = group.dropna(subset=["human_score"])

    if len(graded) < 3:
        group["calibrated_score"] = group["ai_score"]
        group["cal_scale"]        = 1.0
        group["cal_shift"]        = 0.0
        calibration_log[qid]      = {"scale": 1.0, "shift": 0.0, "note": "insufficient_data"}
    else:
        ai_vals = graded["ai_score"].astype(float).values
        hu_vals = graded["human_score"].astype(float).values

        if np.std(ai_vals) < 1e-6:
            shift = float(np.mean(hu_vals) - np.mean(ai_vals))
            scale = 1.0
            note  = "degenerate_input_shift_only"
            print(f"   {qid:8s}  ⚠ all AI scores identical — shift-only calibration applied")
        else:
            try:
                coeffs       = np.polyfit(ai_vals, hu_vals, 1)
                scale, shift = float(coeffs[0]), float(coeffs[1])
                note         = "ok"
            except np.linalg.LinAlgError:
                scale, shift = 1.0, float(np.mean(hu_vals) - np.mean(ai_vals))
                note         = "svd_fallback"
                print(f"   {qid:8s}  ⚠ polyfit SVD failed — shift-only fallback used")

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
        calibration_log[qid] = {
            "scale": round(scale, 4),
            "shift": round(shift, 4),
            "n"    : len(graded),
            "note" : note
        }

        if note == "ok":
            print(f"   {qid:8s}  scale={scale:.3f}  shift={shift:+.3f}  (n={len(graded)})")

    calibrated_rows.append(group)

df_calibrated = pd.concat(calibrated_rows).reset_index(drop=True)
df_calibrated.to_csv(OUTPUT_CALIBRATED, index=False)
print(f"\n✅ Calibrated results saved to '{OUTPUT_CALIBRATED}'")


# =============================================================================
# 8. SAVE FLAGGED SCRIPTS FOR HUMAN REVIEW — UNCHANGED from v3
# =============================================================================

if low_confidence_flags:
    df_flagged = pd.DataFrame(low_confidence_flags)
    df_flagged.to_csv(OUTPUT_FLAGGED, index=False)
    print(f"⚠️  Flagged scripts saved to '{OUTPUT_FLAGGED}'")
else:
    print("✅ No low-confidence scripts to flag.")


# =============================================================================
# 9. EVALUATION METRICS — extended to compare v3 vs v4 if v3 results exist
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

    return {"mae": mae, "rmse": rmse, "r": r,
            "within_1": within_1_mark, "within_5": within_5_marks}


print(f"\n{'='*50}")
print("  📊 EVALUATION RESULTS — v4")
print(f"{'='*50}")

v4_raw_metrics  = print_metrics(df_results,    "ai_score",         "RAW AI Scores (v4)")
v4_cal_metrics  = print_metrics(df_calibrated, "calibrated_score", "CALIBRATED Scores (v4)")

# ── Optional: load v3 results for side-by-side comparison ─────────────────
try:
    df_v3_raw = pd.read_csv("batch_results_v3.csv")
    df_v3_cal = pd.read_csv("batch_results_calibrated.csv")

    print(f"\n{'='*50}")
    print("  📊 COMPARISON: v3 vs v4")
    print(f"{'='*50}")

    def get_metrics(df, score_col):
        ev = df.dropna(subset=["human_score", score_col])
        h  = ev["human_score"].astype(float)
        a  = ev[score_col].astype(float)
        r, _ = pearsonr(h, a)
        return {
            "mae" : mean_absolute_error(h, a),
            "rmse": float(np.sqrt(np.mean((h - a) ** 2))),
            "r"   : r,
            "w5"  : (np.abs(h - a) <= 5).mean() * 100
        }

    v3r = get_metrics(df_v3_raw, "ai_score")
    v3c = get_metrics(df_v3_cal, "calibrated_score")

    def arrow(old, new, lower_is_better=True):
        if lower_is_better:
            return "↓ better" if new < old else ("↑ worse" if new > old else "═ same")
        else:
            return "↑ better" if new > old else ("↓ worse" if new < old else "═ same")

    print(f"\n  {'Metric':<22} {'v3 Raw':>10} {'v4 Raw':>10} {'Change':>12}")
    print(f"  {'─'*56}")
    print(f"  {'MAE':<22} {v3r['mae']:>10.3f} {v4_raw_metrics['mae']:>10.3f} "
          f"  {arrow(v3r['mae'], v4_raw_metrics['mae'])}")
    print(f"  {'RMSE':<22} {v3r['rmse']:>10.3f} {v4_raw_metrics['rmse']:>10.3f} "
          f"  {arrow(v3r['rmse'], v4_raw_metrics['rmse'])}")
    print(f"  {'Pearson r':<22} {v3r['r']:>10.3f} {v4_raw_metrics['r']:>10.3f} "
          f"  {arrow(v3r['r'], v4_raw_metrics['r'], lower_is_better=False)}")
    print(f"  {'Within ±5 marks':<22} {v3r['w5']:>10.1f} {v4_raw_metrics['within_5']:>10.1f} "
          f"  {arrow(v3r['w5'], v4_raw_metrics['within_5'], lower_is_better=False)}")

    print(f"\n  {'Metric':<22} {'v3 Cal':>10} {'v4 Cal':>10} {'Change':>12}")
    print(f"  {'─'*56}")
    print(f"  {'MAE':<22} {v3c['mae']:>10.3f} {v4_cal_metrics['mae']:>10.3f} "
          f"  {arrow(v3c['mae'], v4_cal_metrics['mae'])}")
    print(f"  {'RMSE':<22} {v3c['rmse']:>10.3f} {v4_cal_metrics['rmse']:>10.3f} "
          f"  {arrow(v3c['rmse'], v4_cal_metrics['rmse'])}")
    print(f"  {'Pearson r':<22} {v3c['r']:>10.3f} {v4_cal_metrics['r']:>10.3f} "
          f"  {arrow(v3c['r'], v4_cal_metrics['r'], lower_is_better=False)}")
    print(f"  {'Within ±5 marks':<22} {v3c['w5']:>10.1f} {v4_cal_metrics['within_5']:>10.1f} "
          f"  {arrow(v3c['w5'], v4_cal_metrics['within_5'], lower_is_better=False)}")

except FileNotFoundError:
    print("\n  (v3 results not found — skipping comparison table)")


print(f"\n{'='*50}")
print("  🏁 PIPELINE COMPLETE — v4")
print(f"{'='*50}")
print(f"  Raw scores     → {OUTPUT_RAW}")
print(f"  Calibrated     → {OUTPUT_CALIBRATED}")
print(f"  Flagged        → {OUTPUT_FLAGGED}")
print(f"\n  Calibration log:")
for qid, info in calibration_log.items():
    print(f"    {qid}: {info}")