import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import EmbeddingFunction, Documents, Embeddings
import time
import json
import re
from datetime import datetime
from PIL import Image
import shutil
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AES System v4", layout="wide", page_icon="🎓")

# --- CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

MAX_STUDENT_CHARS   = 3000
LLM_RETRY_ATTEMPTS  = 3


# ==============================================================================
# 🛠️ V4 PREPROCESSING FUNCTIONS
# ==============================================================================

def clean_ocr_text(text: str) -> str:
    """
    FIX 5: Remove OCR diagram artifacts before grading.
    Replaces [[Diagram: description...]] with [student diagram] marker
    so the LLM knows a diagram exists without being confused by
    hundreds of characters of structural description.
    """
    # Replace bracket-enclosed diagram descriptions
    cleaned = re.sub(r'\[\[.*?\]\]', '[student diagram]', text, flags=re.DOTALL)
    # Replace very long parenthetical descriptions (>150 chars) — OCR figure captions
    cleaned = re.sub(r'\(([^)]{150,})\)', '(see diagram)', cleaned)
    return cleaned.strip()


def get_concept_bounds(max_marks: float) -> tuple:
    """
    FIX 1: Tighter concept scaling for high-mark questions.
    Prevents concept inflation where 55-mark questions generate
    13+ concepts and under-score students who cover core ideas.
    """
    if max_marks >= 45:
        # e.g. 55-mark essay: max 6 concepts
        min_c = max(4, int(max_marks / 12))
        max_c = max(6, int(max_marks / 8))
    elif max_marks >= 20:
        # e.g. 24-mark, 45-mark questions
        min_c = max(3, int(max_marks / 10))
        max_c = max(5, int(max_marks / 6))
    else:
        # e.g. 10-mark, 15-mark questions
        min_c = max(2, int(max_marks / 6))
        max_c = max(4, int(max_marks / 4))
    return min_c, max_c


def run_relevance_precheck(student_text: str, question_text: str, max_marks: float):
    """
    FIX 2: Two-stage relevance gate before main LLM grading.

    Returns (0.0, "low", reason_string) to short-circuit grading,
    or None to proceed normally.
    """
    # Gate 1 — too short to be a real answer
    stripped = re.sub(r'\[student diagram\]', '', student_text).strip()
    if len(stripped) < 30:
        return 0.0, "low", "flagged_blank"

    # Gate 2 — fast YES/NO relevance check (only for higher-mark questions)
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
            if resp.text.strip().upper() == "NO":
                return 0.0, "low", "flagged_irrelevant"
        except:
            pass  # if relevance check fails, proceed to full grading

    return None  # proceed normally


# ==============================================================================
# 🧠 SCORING AGENT — V4
# ==============================================================================

class ScoringAgent:
    def __init__(self):
        self.chroma        = chromadb.PersistentClient(path="my_knowledge_base")
        self.embedding_fn  = GoogleEmbeddingFunction()
        self.static_mem    = self.chroma.get_or_create_collection(
            name="lecture_notes", embedding_function=self.embedding_fn)
        self.dynamic_mem   = self.chroma.get_or_create_collection(
            name="grading_memory", embedding_function=self.embedding_fn)
        self.model_answers_mem = self.chroma.get_or_create_collection(
            name="model_answers_cache", embedding_function=self.embedding_fn)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def extract_text_from_image(self, image):
        prompt = """
        You are a research assistant digitizing handwritten student answers.
        INSTRUCTIONS:
        1. Transcribe text exactly as it appears.
        2. EXCLUSION RULE: If text is crossed out/struck through, DO NOT include it.
        3. Ignore teacher marks.
        4. Return raw text only.
        """
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content([prompt, image])
            return response.text.strip()
        except:
            return "Error extracting text."

    # ------------------------------------------------------------------
    # RAG — Reference Answer Generation (unchanged)
    # ------------------------------------------------------------------
    def generate_expected_answer(self, q_text, rubric, max_marks):
        """Generates exam-style reference answer strictly aligned to lecture content."""

        # Cache check
        try:
            results = self.model_answers_mem.query(query_texts=[q_text], n_results=1)
            if results["documents"] and results["distances"][0][0] < 0.05:
                return results["metadatas"][0][0]["answer"]
        except:
            pass

        # Retrieve lecture context
        lecture_context = ""
        try:
            results    = self.static_mem.query(query_texts=[q_text], n_results=3)
            valid_docs = [
                doc for doc, dist in zip(results["documents"][0], results["distances"][0])
                if dist < 0.4
            ]
            lecture_context = "\n---\n".join(valid_docs) if valid_docs else ""
        except:
            pass

        target_words = int(max_marks) * 8
        prompt = f"""
        You are a university student answering an exam question.

        RULES:
        - Use ONLY concepts that appear explicitly in the LECTURE CONTENT
        - Do NOT introduce external theories, laws, or frameworks
          unless explicitly mentioned in the question.
        - Write a concise, marking-scheme style answer.
        - NO FLUFF: Go straight to the point. Start immediately with the answer.

        Question: "{q_text}"
        Max Marks: {max_marks} (~{target_words} words)

        LECTURE CONTENT:
        {lecture_context if lecture_context else "No direct lecture text. Use standard definitions only."}

        Rubric: {rubric if rubric else "Concept accuracy"}

        Write the answer directly. No introductions. No conclusions.
        """
        try:
            model  = genai.GenerativeModel("gemini-2.5-flash")
            resp   = model.generate_content(prompt, generation_config={"temperature": 0.10})
            answer = resp.text.strip()
            if len(answer) > 20:
                import uuid
                self.model_answers_mem.upsert(
                    documents=[q_text],
                    metadatas=[{"answer": answer}],
                    ids=[str(uuid.uuid4())]
                )
            return answer
        except:
            return "Error generating answer."

    # ------------------------------------------------------------------
    # MAIN EVALUATE — V4 pipeline
    # ------------------------------------------------------------------
    def evaluate(self, student_text, q_text, rubric, max_marks, q_id="LIVE"):
        """
        Full v4 grading pipeline:
          1. Clean OCR diagram artifacts (Fix 5)
          2. Relevance pre-check gate (Fix 2)
          3. Generate reference answer via RAG
          4. LLM concept scoring with v4 prompt (Fixes 1, 3, 4, 5)
          5. Return score, confidence, structured feedback
        """

        # ── Step 1: Clean OCR artifacts ────────────────────────────────
        student_clean = clean_ocr_text(student_text)
        has_diagram   = '[student diagram]' in student_clean

        # Truncate very long answers
        if len(student_clean) > MAX_STUDENT_CHARS:
            student_clean = student_clean[:MAX_STUDENT_CHARS] + "\n...[answer truncated for length]"

        # ── Step 2: Relevance pre-check ────────────────────────────────
        precheck = run_relevance_precheck(student_clean, q_text, max_marks)
        if precheck is not None:
            score, confidence, reason = precheck
            return (
                round(score, 1),
                confidence,
                f"⚠️ Answer flagged as {reason.replace('_', ' ')}. Score set to 0 for human review.",
                "N/A — answer flagged before grading",
                []
            )

        # ── Step 3: Generate reference answer via RAG ──────────────────
        expected_answer = self.generate_expected_answer(q_text, rubric, max_marks)

        # ── Step 4: LLM concept scoring ────────────────────────────────
        result = self.get_llm_judgment_v4(
            student_clean, expected_answer, q_text, rubric, max_marks, has_diagram
        )

        score      = float(result.get("score", 0))
        confidence = result.get("confidence", "medium")
        concepts   = result.get("concepts", [])
        feedback   = result.get("feedback", [])

        final_score = round(max(0, min(score, max_marks)), 1)

        return final_score, confidence, feedback, expected_answer, concepts

    # ------------------------------------------------------------------
    # V4 LLM GRADING PROMPT — all fixes applied
    # ------------------------------------------------------------------
    def get_llm_judgment_v4(self, student, expected, question, rubric, max_marks, has_diagram=False):
        """
        V4 prompt with:
          Fix 1: Tighter concept scaling
          Fix 3: Structural knowledge partial credit rule
          Fix 4: Depth verification + domain error + duplication rules
          Fix 5: Diagram-aware grading instructions
        """
        min_concepts, max_concepts = get_concept_bounds(max_marks)

        high_mark_instructions = ""
        if max_marks >= 45:
            high_mark_instructions = f"""
  - IMPORTANT: This is a HIGH-MARK essay question. Focus ONLY on the {max_concepts} most central, essential concepts.
  - Ignore peripheral details, advanced elaborations, and optional examples in the reference.
  - A student who covers the core ideas deserves significant credit even without every detail."""

        diagram_instructions = ""
        if has_diagram:
            diagram_instructions = """
  DIAGRAM RULE (this answer contains student diagrams):
  - The marker [student diagram] indicates a hand-drawn diagram or sketch.
  - If surrounding written text explains what the diagram shows, award full concept credit.
  - If the student wrote a definition and references a diagram without further text,
    award partial credit (0.5) — diagram shows intent but cannot be verified.
  - NEVER give 0 marks solely because an answer is diagram-heavy.
  - Diagram usage is a valid academic choice in handwritten examinations."""

        prompt = f"""
You are a fair and experienced university examiner grading a {max_marks}-mark question.

IMPORTANT CONTEXT:
The REFERENCE ANSWER below was written at an advanced level — it is a MODEL answer,
not a student answer. Students are NOT expected to use the same terminology, phrasing,
or level of detail. Your job is to assess whether the STUDENT UNDERSTANDS the underlying
concept, regardless of how simply or informally they express it.

If a student explains a concept correctly in plain, everyday language, that is FULL CREDIT.
Penalise only for factual incorrectness or genuine conceptual gaps — never for informal wording.

QUESTION:
{question}

REFERENCE ANSWER (use only as a concept map — not a wording benchmark):
{expected}

STUDENT ANSWER:
{student}

RUBRIC:
{rubric if rubric and rubric.strip() not in ("", "nan") else "Conceptual understanding and correctness."}

=== GRADING INSTRUCTIONS ===

STEP 1 — BUILD A CONCEPT MAP from the REFERENCE ANSWER.
  - Distil into core ideas that answer the QUESTION.
  - Strip away advanced vocabulary — ask: "what is the underlying idea here?"
  - Target between {min_concepts} and {max_concepts} concepts.
  - Each concept should be expressible in one plain sentence.{high_mark_instructions}

STEP 2 — CHECK STUDENT UNDERSTANDING for each concept.

  CREDIT LEVELS:
  - "full"    (1.0): Student demonstrates clear understanding, even in simple words.
  - "partial" (0.5): Some awareness but incomplete, vague, or slightly imprecise.
  - "missing" (0.0): No understanding, factually incorrect, or not addressed.

  DEPTH VERIFICATION RULE:
  Before awarding "full" credit, ask: does the student show WHY or HOW the concept works?
  - Names concept only → "partial" at most
  - Names + thin/wrong explanation → "partial"
  - Names + correct explanation → "full"

  STRUCTURAL KNOWLEDGE RULE:
  If a student correctly lists, names, or enumerates components/steps even without
  explaining each one — award "partial" (0.5) minimum per correctly identified item.
  NEVER give "missing" for a correctly named but unexplained concept.

  DOMAIN ERROR RULE:
  If a student's example is from a clearly wrong domain (e.g., beverage company when
  question is about a supermarket POS), downgrade from "full" to "partial".

  DUPLICATION RULE:
  If the same step, concept, or idea appears more than once, count it ONCE only.
{diagram_instructions}

STEP 3 — COMPUTE SCORE:
  credited = sum of credits across all concepts
  score    = (credited / total_concepts) × {max_marks}
  Round to nearest 0.25.

CRITICAL RULES:
  - NEVER penalise for not using technical terminology.
  - NEVER penalise for brevity if core understanding is present.
  - A completely blank or entirely off-topic answer scores 0.
  - Do NOT reward padding or repetition.

STEP 4 — ASSIGN CONFIDENCE:
  - "high"   → concepts clear, matching straightforward
  - "medium" → some ambiguity or borderline cases
  - "low"    → answer unreadable, mainly diagrams, or genuinely unclear

Return ONLY valid JSON (no markdown, no extra text):
{{
  "concepts": [
    {{
      "concept": "<plain-language description>",
      "status": "full|partial|missing",
      "student_evidence": "<brief quote or paraphrase, or 'nothing'>"
    }}
  ],
  "total_concepts": <integer>,
  "credited_concepts": <float>,
  "score": <float>,
  "confidence": "high|medium|low",
  "feedback": ["<bullet point 1>", "<bullet point 2>", "<bullet point 3>"]
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
                clean = response.text.strip().replace("```json", "").replace("```", "").strip()
                data  = json.loads(clean)
                return {
                    "score"     : float(data.get("score", 0)),
                    "confidence": str(data.get("confidence", "medium")).lower(),
                    "concepts"  : data.get("concepts", []),
                    "feedback"  : data.get("feedback", [])
                }
            except Exception as e:
                if attempt < LLM_RETRY_ATTEMPTS:
                    time.sleep(2.0 * attempt)

        return {"score": 0.0, "confidence": "low", "concepts": [], "feedback": ["Grading error — please retry."]}

    # ------------------------------------------------------------------
    # LEARN — unchanged
    # ------------------------------------------------------------------
    def learn(self, text, score, max_marks, feedback, q_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dynamic_mem.upsert(
            documents=[text],
            metadatas=[{
                "question_id"    : q_id,
                "score"          : str(score),
                "max_marks"      : str(max_marks),
                "teacher_feedback": feedback,
                "type"           : "human_override"
            }],
            ids=[f"learn_{q_id}_{timestamp}"]
        )


# --- EMBEDDING HELPER ---
def get_embedding(text):
    if not isinstance(text, str) or len(text) < 2:
        return None
    try:
        resp = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="semantic_similarity"
        )
        vec = np.array(resp['embedding'])
        if np.count_nonzero(vec) == 0:
            return None
        return vec.reshape(1, -1)
    except:
        return None

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = [get_embedding(t) for t in input]
        return [e[0].tolist() if e is not None else np.random.rand(768).tolist() for e in embeddings]


# ==============================================================================
# 🖥️  UI
# ==============================================================================
st.title("🎓 Multi-Agent Adaptive Grading System — v4")

# ── Confidence badge helper ────────────────────────────────────────────────
def confidence_badge(conf):
    if conf == "high":
        return "🟢 High Confidence"
    elif conf == "medium":
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence — Recommend Human Review"

# ── Factory Reset ──────────────────────────────────────────────────────────
def factory_reset():
    if os.path.exists("my_knowledge_base"):
        shutil.rmtree("my_knowledge_base")
    st.cache_resource.clear()
    st.session_state.clear()
    return True

with st.sidebar:
    st.header("⚙️ System Tools")
    st.markdown("**Grading Method:** LLM Concept v4")
    st.markdown("**Fixes Applied:**")
    st.markdown("- ✅ Tight concept scaling")
    st.markdown("- ✅ Relevance pre-check gate")
    st.markdown("- ✅ Structural knowledge credit")
    st.markdown("- ✅ Depth + domain + dupe rules")
    st.markdown("- ✅ OCR diagram cleaning")
    st.divider()
    st.warning("⚠️ Use reset if AI is confused.")
    if st.button("☢️ FACTORY RESET (Wipe All Memory)", type="primary"):
        if factory_reset():
            st.success("System Wiped! Reloading...")
            time.sleep(1)
            st.rerun()

# ── Init ───────────────────────────────────────────────────────────────────
if "scorer" not in st.session_state:
    st.session_state.scorer = ScoringAgent()

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

tab1, tab2 = st.tabs(["🔴 Live Grading", "📂 Batch Processing"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE GRADING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("1. Assignment Details")
        st.caption("💡 Use a new Question ID for each unique question.")
        q_id      = st.text_input("Question ID", value="Q_Live_01")
        q_text    = st.text_area("Question Text", value="Explain Usability Engineering.")
        max_marks = st.number_input("Max Marks", value=50, min_value=1)
        rubric    = st.text_area("Marking Scheme (Optional)", value="")

        st.divider()
        st.subheader("2. Student Submission")

        uploaded_img = st.file_uploader(
            "📸 Upload Handwritten Script (Optional)",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_img:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Script", use_column_width=True)
            if st.button("📝 Extract Text from Image"):
                with st.spinner("👀 Reading handwriting..."):
                    raw_text = st.session_state.scorer.extract_text_from_image(image)
                    st.session_state.extracted_text = raw_text
                    st.success("✅ Text extracted!")
                    # Show if diagrams were detected
                    cleaned = clean_ocr_text(raw_text)
                    if '[student diagram]' in cleaned:
                        st.info("📐 Diagram detected in answer — diagram-aware grading will be applied.")

        student_input = st.text_area(
            "Answer Text (edit if needed)",
            value=st.session_state.extracted_text,
            height=220,
            key="student_input_area"
        )

        if st.button("🚀 Grade Submission", type="primary"):
            if not student_input.strip():
                st.warning("Please enter or upload a student answer first.")
            else:
                with st.spinner("🤖 Evaluating with v4 scoring..."):
                    score, confidence, feedback, expected_ans, concepts = \
                        st.session_state.scorer.evaluate(
                            student_input, q_text, rubric, float(max_marks), q_id
                        )
                    st.session_state.last_result = {
                        "score"     : score,
                        "confidence": confidence,
                        "feedback"  : feedback,
                        "text"      : student_input,
                        "expected"  : expected_ans,
                        "concepts"  : concepts
                    }

    # ── Results Panel ───────────────────────────────────────────────────────
    with col_result:
        st.subheader("3. Result")

        if "last_result" in st.session_state:
            res = st.session_state.last_result

            # Reference answer
            st.info("🧠 Reference Answer Used (RAG-Generated)")
            with st.expander("View Reference Answer"):
                st.write(res['expected'])

            # Score + confidence
            col_s, col_c = st.columns(2)
            with col_s:
                st.metric("Score", f"{res['score']} / {max_marks}")
            with col_c:
                conf = res['confidence']
                if conf == "high":
                    st.success(confidence_badge(conf))
                elif conf == "medium":
                    st.warning(confidence_badge(conf))
                else:
                    st.error(confidence_badge(conf))

            # Feedback
            st.markdown("### 📋 Feedback")
            if isinstance(res['feedback'], list) and res['feedback']:
                for item in res['feedback']:
                    st.markdown(f"- {item}")
            else:
                st.write(res['feedback'])

            # Concept breakdown
            if res['concepts']:
                st.markdown("### 🔍 Concept Breakdown")
                concept_rows = []
                for c in res['concepts']:
                    status = c.get('status', '')
                    icon   = "✅" if status == "full" else ("⚠️" if status == "partial" else "❌")
                    concept_rows.append({
                        "Concept"         : c.get('concept', ''),
                        "Status"          : f"{icon} {status}",
                        "Student Evidence": c.get('student_evidence', '')
                    })
                st.dataframe(
                    pd.DataFrame(concept_rows),
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            # Override
            st.markdown("### ✏️ Lecturer Override")
            st.caption("Disagree with the score? Override and teach the system.")
            new_score     = st.number_input(
                "Override Score",
                value=float(res['score']),
                min_value=0.0,
                max_value=float(max_marks),
                step=0.5,
                key="override_score"
            )
            teacher_notes = st.text_input("Reason for Override", key="override_notes")
            if st.button("💾 Save Override & Learn"):
                st.session_state.scorer.learn(
                    res['text'], new_score, max_marks, teacher_notes, q_id
                )
                st.success(f"✅ System updated: score {res['score']} → {new_score} saved.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📂 Batch Processing")
    st.markdown("Upload a CSV with columns: `question_id`, `question_text`, `student_answer`, `max_marks`, `marking_scheme` (optional), `teacher_score` (optional for evaluation).")

    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")

    if uploaded_csv:
        df_batch = pd.read_csv(uploaded_csv)
        st.write(f"**Preview ({len(df_batch)} rows):**")
        st.dataframe(df_batch.head(5), use_container_width=True)

        # Validate required columns
        required = {"question_id", "question_text", "student_answer", "max_marks"}
        if not required.issubset(set(df_batch.columns)):
            st.error(f"CSV missing required columns. Need: {required}")
        else:
            if st.button("🚀 Run Batch Grading", type="primary"):
                results   = []
                progress  = st.progress(0)
                status_box = st.empty()

                for i, row in df_batch.iterrows():
                    qid          = str(row["question_id"])
                    q_text_b     = str(row["question_text"])
                    student_ans  = str(row["student_answer"])
                    max_marks_b  = float(row["max_marks"])
                    rubric_b     = str(row.get("marking_scheme", ""))

                    status_box.markdown(f"**Grading [{i+1}/{len(df_batch)}]:** {qid}...")

                    score, confidence, feedback, expected, concepts = \
                        st.session_state.scorer.evaluate(
                            student_ans, q_text_b, rubric_b, max_marks_b, qid
                        )

                    result_row = {
                        "question_id" : qid,
                        "ai_score"    : score,
                        "max_marks"   : max_marks_b,
                        "confidence"  : confidence,
                        "num_concepts": len(concepts),
                        "feedback"    : " | ".join(feedback) if isinstance(feedback, list) else str(feedback)
                    }
                    if "teacher_score" in df_batch.columns:
                        result_row["teacher_score"] = row.get("teacher_score", "")

                    results.append(result_row)
                    progress.progress((i + 1) / len(df_batch))
                    time.sleep(1.0)

                progress.empty()
                status_box.empty()

                df_results = pd.DataFrame(results)
                st.success(f"✅ Batch grading complete — {len(df_results)} scripts graded.")
                st.dataframe(df_results, use_container_width=True)

                # Evaluation metrics if teacher scores are present
                if "teacher_score" in df_results.columns:
                    eval_df = df_results.dropna(subset=["teacher_score", "ai_score"])
                    if len(eval_df) >= 3:
                        from sklearn.metrics import mean_absolute_error
                        from scipy.stats import pearsonr
                        h   = eval_df["teacher_score"].astype(float)
                        a   = eval_df["ai_score"].astype(float)
                        mae = mean_absolute_error(h, a)
                        rmse = float(np.sqrt(np.mean((h - a) ** 2)))
                        r, _ = pearsonr(h, a)
                        st.markdown("### 📊 Evaluation Metrics")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("MAE", f"{mae:.3f} marks")
                        mc2.metric("RMSE", f"{rmse:.3f}")
                        mc3.metric("Pearson r", f"{r:.3f}")

                # Download button
                csv_out = df_results.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Results CSV",
                    data=csv_out,
                    file_name=f"batch_results_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )