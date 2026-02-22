
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
import time
import json
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import EmbeddingFunction, Documents, Embeddings

# --- 1. CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

# --- 2. SETUP MEMORIES ---
print("🔌 Connecting to Hybrid Intelligence System...")
chroma_client = chromadb.PersistentClient(path="my_knowledge_base")

def get_embedding(text):
    if not isinstance(text, str) or len(text) < 2: return np.zeros(768)
    try:
        resp = genai.embed_content(model="models/text-embedding-004", content=text, task_type="semantic_similarity")
        return np.array(resp['embedding'])
    except: return np.zeros(768)

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [get_embedding(t).tolist() for t in input]

# Brain 1: Static (Lectures) - CRITICAL for Context-Awareness
try:
    static_mem = chroma_client.get_collection(name="lecture_notes", embedding_function=GoogleEmbeddingFunction())
except:
    static_mem = None

# Brain 2: Dynamic (Experience) - Disabled for Batch Test to prevent Data Leakage
dynamic_mem = None

def get_llm_score(student_ans, expected_ans, max_marks, question, rubric):
    prompt = f"""
        You are a strict but fair university examiner.

        Your grading must be mathematically proportional.

        STEP 1:
        From the REFERENCE ANSWER, extract only the MAIN CONCEPTS that are directly relevant to the QUESTION.
        Ignore minor details or additional information that is not critical to answering the question.
        Each main concept = 1 concept.

        STEP 2:
        Check which concepts are correctly present in the student answer
        (even if phrased differently).

        STEP 3:
        Compute:

        score = (number_of_correct_concepts / total_concepts) * {max_marks}

        RULES:
        - Do NOT penalize brevity.
        - Do NOT reward extra writing.
        - Do NOT compare wording.
        - Partial concept = half credit for that concept.
        - If student is completely wrong, score near 0.

        QUESTION:
        {question}

        REFERENCE ANSWER:
        {expected_ans}

        STUDENT ANSWER:
        {student_ans}

        RUBRIC:
        {rubric if rubric else "Concept correctness and proportional scoring"}

        Return ONLY JSON:
        {{
        "total_concepts": number,
        "matched_concepts": number,
        "score": number
        }}
        """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        )

        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_text)

        return float(data["score"])

    except Exception as e:
        print("LLM error:", e)
        return 0


# --- 4. LOAD DATA ---
print("📂 Loading Datasets...")
# Use .tail(6) for testing, remove for full run
df_students = pd.read_csv("cleaned_dataset.csv")
df_expected = pd.read_csv("expected_answers_dual.csv")

# Merge logic
common_cols = [col for col in df_students.columns if col in df_expected.columns and col != 'question_id']
df_students = df_students.drop(columns=common_cols)
df_merged = pd.merge(df_students, df_expected, on='question_id', how='left')

# Fix column names
if 'marking_scheme_y' in df_merged.columns: df_merged.rename(columns={'marking_scheme_y': 'marking_scheme'}, inplace=True)
if 'max_marks_y' in df_merged.columns: df_merged.rename(columns={'max_marks_y': 'max_marks'}, inplace=True)

print(f"--- 🤖 Gated Hybrid Grading of {len(df_merged)} Scripts ---")
final_results = []

# --- 5. HYBRID GRADING LOOP ---
for index, row in df_merged.iterrows():
    qid = row['question_id']
    student_text = str(row['extracted_text'])
    q_text = str(row['question_text'])
    rubric = str(row['marking_scheme']) if 'marking_scheme' in row else ""
    
    try: max_marks = float(row['max_marks'])
    except: max_marks = 10.0
    
    print(f"   Processing {qid}...", end="")

    expected_essay = str(row.get("expected_essay", ""))
    expected_visual = str(row.get("expected_visual", ""))

    v_q = get_embedding(q_text)
    essay_score = 0
    visual_score = 0

    if v_q is not None:
        if expected_essay:
            v_e = get_embedding(expected_essay)
            if v_e is not None:
                essay_score = cosine_similarity(
                    v_e.reshape(1, -1), v_q.reshape(1, -1)
                )[0][0]

        if expected_visual:
            v_v = get_embedding(expected_visual)
            if v_v is not None:
                visual_score = cosine_similarity(
                    v_v.reshape(1, -1), v_q.reshape(1, -1)
                )[0][0]

    expected_answer = (
        expected_essay if essay_score >= visual_score else expected_visual
    )
    
    relevance_penalty = 0
    v_student = get_embedding(student_text)

    if v_student is not None and v_q is not None:
        sim_sq = cosine_similarity(
            v_student.reshape(1, -1),
            v_q.reshape(1, -1)
        )[0][0]

        # if sim_sq < 0.3:
        #     relevance_penalty = 0.2 * max_marks

        llm_result = get_llm_score(
        student_text,
        expected_answer,
        max_marks,
        q_text,
        rubric,

    )

    llm_score = float(llm_result)

    fail_threshold = 0.4 * max_marks

    if llm_score < fail_threshold:
        final_score = llm_score
    else:
        final_score = max(llm_score - relevance_penalty, 0)

    final_score = min(final_score, max_marks)

    final_results.append({
        "question_id": qid,
        "human_score": row['teacher_score'],
        "ai_score": round(final_score, 2),
        "max_marks": max_marks,
        "method": "LLM Concept + Relevance Penalty",
    })
    
    time.sleep(1.0)
    print(" Done.")

# --- 6. SAVE ---
output_file = "batch_results_live_aligned2.csv"
pd.DataFrame(final_results).to_csv(output_file, index=False)
print("\n" + "="*50)
print(f"✅ BATCH COMPLETE! Saved to '{output_file}'")
print("="*50)