import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import EmbeddingFunction, Documents, Embeddings

# --- 1. CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

# THEORETICAL WEIGHTS (Formula)
W_SEMANTIC = 0.5
W_RUBRIC   = 0.3
W_CONTEXT  = 0.2

# ADAPTIVE WEIGHT (How much to trust history)
HISTORY_CONFIDENCE_THRESHOLD = 0.85 
HISTORY_WEIGHT = 0.4 

# --- 2. SETUP MEMORIES ---
print("🔌 Connecting to Double-Brain System...")
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
    print("⚠️ Warning: Static memory not found. Context scoring will be skipped.")
    static_mem = None

# Brain 2: Dynamic (Experience)
try:
    dynamic_mem = chroma_client.get_collection(name="grading_memory", embedding_function=GoogleEmbeddingFunction())
except:
    print("⚠️ Warning: Dynamic memory not found. Adaptive scoring will be skipped.")
    dynamic_mem = None

try:
    chroma_client.delete_collection("grading_memory")
    print("🧹 Wiped Grading Memory to force fresh calculation.")
except: pass

# --- 3. LOAD DATA (ROBUST FIX) ---
print("📂 Loading Datasets...")

# Load files
df_students = pd.read_csv("cleaned_dataset.csv").tail(6)
df_expected = pd.read_csv("expected_answers_dual.csv")

# Identify duplicated columns that might cause conflict (excluding the key 'question_id')
common_cols = [col for col in df_students.columns if col in df_expected.columns and col != 'question_id']

# Drop these from students df to prioritize the "Gold Standard" columns from expected_answers
df_students = df_students.drop(columns=common_cols)

# Merge
df_merged = pd.merge(df_students, df_expected, on='question_id', how='left')

# Safety Check: Ensure 'marking_scheme' exists
if 'marking_scheme' not in df_merged.columns:
    print("⚠️ Column 'marking_scheme' missing. Checking for suffixes...")
    if 'marking_scheme_y' in df_merged.columns:
        df_merged.rename(columns={'marking_scheme_y': 'marking_scheme'}, inplace=True)
    elif 'marking_scheme_x' in df_merged.columns:
        df_merged.rename(columns={'marking_scheme_x': 'marking_scheme'}, inplace=True)
    else:
        print("❌ Critical Error: marking_scheme not found in either file.")
        exit()

# Safety Check: Ensure 'max_marks' exists
if 'max_marks' not in df_merged.columns:
    if 'max_marks_y' in df_merged.columns:
        df_merged.rename(columns={'max_marks_y': 'max_marks'}, inplace=True)
    elif 'max_marks_x' in df_merged.columns:
        df_merged.rename(columns={'max_marks_x': 'max_marks'}, inplace=True)

print(f"--- 🧮 Adaptive Grading of {len(df_merged)} Scripts ---")
final_results = []

# --- 4. GRADING LOOP ---
for index, row in df_merged.iterrows():
    qid = row['question_id']
    student_text = str(row['extracted_text'])
    
    # Handle missing max_marks safely
    try:
        max_marks = float(row['max_marks'])
    except:
        max_marks = 10.0 # Fallback
    
    print(f"   Processing {qid}...", end="")

    # Generate embeddings
    v_student = get_embedding(student_text).reshape(1, -1)
    v_essay   = get_embedding(str(row['expected_essay'])).reshape(1, -1)
    v_visual  = get_embedding(str(row['expected_visual'])).reshape(1, -1)
    v_rubric  = get_embedding(str(row['marking_scheme'])).reshape(1, -1)

    # --- PART A: THEORETICAL SCORE ---
    sim_essay = cosine_similarity(v_student, v_essay)[0][0]
    sim_visual = cosine_similarity(v_student, v_visual)[0][0]
    sim_rubric = cosine_similarity(v_student, v_rubric)[0][0]
    
    # Context Check (Static Memory)
    sim_context = 0
    if static_mem:
        try:
            rag_res = static_mem.query(query_texts=[student_text], n_results=1)
            if rag_res['documents'] and rag_res['documents'][0]:
                v_ctx = get_embedding(rag_res['documents'][0][0]).reshape(1, -1)
                sim_context = cosine_similarity(v_student, v_ctx)[0][0]
        except: pass

    # Calculate Raw Formula Ratio (0.0 to 1.0)
    # Max-Pooling: Take the best of Essay vs Visual
    raw_ratio = (max(sim_essay, sim_visual) * W_SEMANTIC) + (sim_rubric * W_RUBRIC) + (sim_context * W_CONTEXT)

    # --- PART B: HISTORICAL CHECK ---
    sim_history = 0
    past_score_ratio = 0
    
    if dynamic_mem:
        try:
            hist_res = dynamic_mem.query(
                query_texts=[student_text], 
                n_results=1,
                where={"question_id": qid} 
            )
            
            if hist_res['documents'] and hist_res['documents'][0]:
                past_text = hist_res['documents'][0][0]
                v_past = get_embedding(past_text).reshape(1, -1)
                sim_history = cosine_similarity(v_student, v_past)[0][0]
                
                # Retrieve past score
                past_score_val = float(hist_res['metadatas'][0][0]['score'])
                past_score_ratio = past_score_val / max_marks
        except: pass

    # --- PART C: THE ADAPTIVE BLEND ---
    final_ratio = raw_ratio 
    
    if sim_history > HISTORY_CONFIDENCE_THRESHOLD:
        # Blend: 60% Formula, 40% History
        final_ratio = ((1 - HISTORY_WEIGHT) * raw_ratio) + (HISTORY_WEIGHT * past_score_ratio)
        print(f" [History Nudge] ", end="")
    
    # Final Marks Calculation
    final_mark = final_ratio * max_marks
    final_mark = max(0, min(final_mark, max_marks)) # Clip between 0 and Max

    final_results.append({
        "question_id": qid,
        "human_score": row['teacher_score'],
        "ai_score": round(final_mark, 2),
        "max_marks": max_marks,
        "history_used": 1 if sim_history > HISTORY_CONFIDENCE_THRESHOLD else 0
    })
    print(" Done.")

# --- 5. SAVE ---
output_file = "adaptive_results.csv"
pd.DataFrame(final_results).to_csv(output_file, index=False)
print("\n" + "="*50)
print(f"✅ CALCULATION COMPLETE! Saved to '{output_file}'")
print("="*50)