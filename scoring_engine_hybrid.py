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

# HYBRID WEIGHTS
W_LLM_JUDGE = 0.8   # 80% Trust in Gemini's Reasoning
W_VECTOR_MATH = 0.2 # 20% Trust in Vector Math (Safety Anchor)

# ADAPTIVE MEMORY SETTINGS
HISTORY_CONFIDENCE_THRESHOLD = 0.85 
HISTORY_WEIGHT = 1.0 # If history matches, we fully trust the human precedent

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

# Brain 1: Static (Lectures)
try:
    static_mem = chroma_client.get_collection(name="lecture_notes", embedding_function=GoogleEmbeddingFunction())
except:
    static_mem = None

# Brain 2: Dynamic (Experience)
# try:
#     dynamic_mem = chroma_client.get_collection(name="grading_memory", embedding_function=GoogleEmbeddingFunction())
# except:
#     dynamic_mem = None

dynamic_mem = None

# --- 3. HELPER: THE LLM JUDGE ---
def get_llm_score(student_ans, expected_ans, max_marks, question):
    """Asks Gemini to act as a strict examiner"""
    prompt = f"""
    You are a strict University Examiner. Grade this student answer.
    
    Question: "{question}"
    Max Marks: {max_marks}
    
    Reference Answer (Gold Standard): "{expected_ans}"
    Student Answer: "{student_ans}"
    
    CRITICAL RULES:
    1. Compare the Student Answer to the Reference Answer.
    2. Focus on MEANING, not just keywords.
    3. If the answer is hallucinated or wrong context, score 0.
    4. Return JSON: {{"score": <number>}}
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json", "temperature": 0.0}
        )
        data = json.loads(response.text)
        return float(data['score'])
    except:
        return -1 # Error flag

# --- 4. LOAD DATA ---
print("📂 Loading Datasets...")
df_students = pd.read_csv("cleaned_dataset.csv")
df_expected = pd.read_csv("expected_answers_dual.csv")

# Merge logic (Same as your math script)
common_cols = [col for col in df_students.columns if col in df_expected.columns and col != 'question_id']
df_students = df_students.drop(columns=common_cols)
df_merged = pd.merge(df_students, df_expected, on='question_id', how='left')

# Fix column names if needed
if 'marking_scheme_y' in df_merged.columns: df_merged.rename(columns={'marking_scheme_y': 'marking_scheme'}, inplace=True)
if 'max_marks_y' in df_merged.columns: df_merged.rename(columns={'max_marks_y': 'max_marks'}, inplace=True)

print(f"--- 🤖 Hybrid Grading of {len(df_merged)} Scripts ---")
final_results = []

# --- 5. HYBRID GRADING LOOP ---
for index, row in df_merged.iterrows():
    qid = row['question_id']
    student_text = str(row['extracted_text'])
    q_text = str(row['question_text'])
    
    try: max_marks = float(row['max_marks'])
    except: max_marks = 10.0
    
    print(f"   Processing {qid}...", end="")

    # --- STEP A: VECTOR MATH SCORE (20% Weight) ---
    v_student = get_embedding(student_text).reshape(1, -1)
    v_essay   = get_embedding(str(row['expected_essay'])).reshape(1, -1)
    v_visual  = get_embedding(str(row['expected_visual'])).reshape(1, -1)
    
    # Similarity Calculation
    sim_essay = cosine_similarity(v_student, v_essay)[0][0]
    sim_visual = cosine_similarity(v_student, v_visual)[0][0]
    
    # RAG Context Check
    sim_context = 0
    if static_mem:
        try:
            rag_res = static_mem.query(query_texts=[student_text], n_results=1)
            if rag_res['documents']:
                v_ctx = get_embedding(rag_res['documents'][0][0]).reshape(1, -1)
                sim_context = cosine_similarity(v_student, v_ctx)[0][0]
        except: pass

    # Math Score (Max of Essay/Visual + Context)
    math_ratio = (max(sim_essay, sim_visual) * 0.7) + (sim_context * 0.3)
    math_score = math_ratio * max_marks

    # --- STEP B: LLM JUDGE SCORE (80% Weight) ---
    # We pass the "Essay" version as the reference for the LLM
    llm_score = get_llm_score(student_text, str(row['expected_essay']), max_marks, q_text)
    
    # Fallback: If API fails, trust Vector Math 100%
    if llm_score == -1:
        print(" [API Error, using Math] ", end="")
        hybrid_score = math_score
    else:
        # THE HYBRID FORMULA
        hybrid_score = (llm_score * W_LLM_JUDGE) + (math_score * W_VECTOR_MATH)

    # --- STEP C: ADAPTIVE MEMORY (Override) ---
    # sim_history = 0
    # past_score_val = 0
    # history_triggered = False
    
    # if dynamic_mem:
    #     try:
    #         hist_res = dynamic_mem.query(
    #             query_texts=[student_text], n_results=1, where={"question_id": qid}
    #         )
    #         if hist_res['documents'] and hist_res['documents'][0]:
    #             v_past = get_embedding(hist_res['documents'][0][0]).reshape(1, -1)
    #             sim_history = cosine_similarity(v_student, v_past)[0][0]
                
    #             if sim_history > HISTORY_CONFIDENCE_THRESHOLD:
    #                 past_score_val = float(hist_res['metadatas'][0][0]['score'])
    #                 history_triggered = True
    #                 print(f" [Memory Used] ", end="")
    #     except: pass

    # # Apply History Override
    # if history_triggered:
    #     final_mark = past_score_val
    # else:
    #     final_mark = hybrid_score

    final_mark = hybrid_score

    # Clip Score
    final_mark = max(0, min(final_mark, max_marks))

    final_results.append({
        "question_id": qid,
        "human_score": row['teacher_score'],
        "ai_score": round(final_mark, 2),
        "max_marks": max_marks,
        "method": "Hybrid (Vector + LLM)"
    })
    
    # Sleep to respect API limits
    time.sleep(1.0)
    print(" Done.")

# --- 6. SAVE ---
output_file = "hybrid_results.csv"
pd.DataFrame(final_results).to_csv(output_file, index=False)
print("\n" + "="*50)
print(f"✅ HYBRID BATCH COMPLETE! Saved to '{output_file}'")
print("="*50)