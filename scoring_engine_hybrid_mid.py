# import pandas as pd
# import numpy as np
# import chromadb
# import google.generativeai as genai
# import time
# import json
# from sklearn.metrics.pairwise import cosine_similarity
# from chromadb import EmbeddingFunction, Documents, Embeddings

# # --- 1. CONFIGURATION ---
# GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
# genai.configure(api_key=GENAI_API_KEY)

# # HYBRID WEIGHTS
# W_LLM_JUDGE = 0.6   # 80% Trust in Gemini's Reasoning
# W_VECTOR_MATH = 0.4 # 20% Trust in Vector Math (Safety Anchor)

# # ADAPTIVE MEMORY SETTINGS
# HISTORY_CONFIDENCE_THRESHOLD = 0.85 
# HISTORY_WEIGHT = 1.0 # If history matches, we fully trust the human precedent

# # --- 2. SETUP MEMORIES ---
# print("🔌 Connecting to Hybrid Intelligence System...")
# chroma_client = chromadb.PersistentClient(path="my_knowledge_base")

# def get_embedding(text):
#     if not isinstance(text, str) or len(text) < 2: return np.zeros(768)
#     try:
#         resp = genai.embed_content(model="models/text-embedding-004", content=text, task_type="semantic_similarity")
#         return np.array(resp['embedding'])
#     except: return np.zeros(768)

# class GoogleEmbeddingFunction(EmbeddingFunction):
#     def __call__(self, input: Documents) -> Embeddings:
#         return [get_embedding(t).tolist() for t in input]

# # Brain 1: Static (Lectures)
# try:
#     static_mem = chroma_client.get_collection(name="lecture_notes", embedding_function=GoogleEmbeddingFunction())
# except:
#     static_mem = None
# dynamic_mem = None

# # --- 3. HELPER: THE LLM JUDGE (CORRECTED) ---
# def get_llm_score(student_ans, expected_ans, max_marks, question):
#     """Asks Gemini to act as a examiner"""
#     # FIX: Added the variables into the string so Gemini actually sees them!
#     prompt = f"""
#     You are a University Examiner.

#     rading Task:
#     - Question: "{question}"
#     - Max Marks: {max_marks}
#     - Reference Answer: "{expected_ans}"
#     - Student Answer: "{student_ans}"

#     Rules:
#     1. Focus on conceptual correctness and reasoning.
#     2. Penalize missing points proportionally.
#     3. If the answer is mostly irrelevant or incorrect, give a very low score (0).
#     4. Do NOT give full marks unless all key points are covered.
#     5. Return JSON only: {{"score": <number_out_of_{max_marks}>}}
#     """
#     try:
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         response = model.generate_content(
#             prompt, 
#             generation_config={"response_mime_type": "application/json", "temperature": 0.0}
#         )
#         data = json.loads(response.text)
#         return float(data['score'])
#     except:
#         print(" [LLM Error] ", end="")
#         return -1 # Error flag

# # --- 4. LOAD DATA ---
# print("📂 Loading Datasets...")
# df_students = pd.read_csv("cleaned_dataset.csv").tail(6)
# df_expected = pd.read_csv("expected_answers_dual.csv")

# # Merge logic (Same as your math script)
# common_cols = [col for col in df_students.columns if col in df_expected.columns and col != 'question_id']
# df_students = df_students.drop(columns=common_cols)
# df_merged = pd.merge(df_students, df_expected, on='question_id', how='left')

# # Fix column names if needed
# if 'marking_scheme_y' in df_merged.columns: df_merged.rename(columns={'marking_scheme_y': 'marking_scheme'}, inplace=True)
# if 'max_marks_y' in df_merged.columns: df_merged.rename(columns={'max_marks_y': 'max_marks'}, inplace=True)

# print(f"--- 🤖 Hybrid Grading of {len(df_merged)} Scripts ---")
# final_results = []

# # --- 5. HYBRID GRADING LOOP ---
# for index, row in df_merged.iterrows():
#     qid = row['question_id']
#     student_text = str(row['extracted_text'])
#     q_text = str(row['question_text'])
    
#     try: max_marks = float(row['max_marks'])
#     except: max_marks = 10.0
    
#     print(f"   Processing {qid}...", end="")

#     # --- STEP A: VECTOR MATH SCORE (40% Weight) ---
#     v_student = get_embedding(student_text).reshape(1, -1)
#     v_essay   = get_embedding(str(row['expected_essay'])).reshape(1, -1)
#     v_visual  = get_embedding(str(row['expected_visual'])).reshape(1, -1)
    
#     # Similarity Calculation
#     sim_essay = cosine_similarity(v_student, v_essay)[0][0]
#     sim_visual = cosine_similarity(v_student, v_visual)[0][0]
    
#     # RAG Context Check
#     sim_context = 0
#     if static_mem:
#         try:
#             rag_res = static_mem.query(query_texts=[student_text], n_results=1)
#             if rag_res['documents']:
#                 v_ctx = get_embedding(rag_res['documents'][0][0]).reshape(1, -1)
#                 sim_context = cosine_similarity(v_student, v_ctx)[0][0]
#         except: pass

#     # Math Score (Max of Essay/Visual + Context)
#     math_ratio = (max(sim_essay, sim_visual) * 0.7) + (sim_context * 0.3)
#     math_score = math_ratio * max_marks

#     # --- STEP B: LLM JUDGE SCORE (60% Weight) ---
#     # We pass the "Essay" version as the reference for the LLM
#     llm_score = get_llm_score(student_text, str(row['expected_essay']), max_marks, q_text)
    
#     # Fallback: If API fails, trust Vector Math 100%
#     if llm_score == -1:
#         print(" [API Error, using Math] ", end="")
#         hybrid_score = math_score
#     # else:
#     #     # THE HYBRID FORMULA
#     #     hybrid_score = (llm_score * W_LLM_JUDGE) + (math_score * W_VECTOR_MATH)
#     elif llm_score < 0.2 * max_marks:
#         hybrid_score = llm_score
#     else:
#         hybrid_score = (llm_score * W_LLM_JUDGE) + (math_score * W_VECTOR_MATH)


#     final_mark = hybrid_score

#     # Clip Score
#     final_mark = max(0, min(final_mark, max_marks))

#     final_results.append({
#         "question_id": qid,
#         "human_score": row['teacher_score'],
#         "ai_score": round(final_mark, 2),
#         "max_marks": max_marks,
#         "method": "Hybrid (Vector 40% + LLM 60%)"
#     })
    
#     # Sleep to respect API limits
#     time.sleep(1.0)
#     print(" Done.")

# # --- 6. SAVE ---
# output_file = "hybrid_results_part2.csv"
# pd.DataFrame(final_results).to_csv(output_file, index=False)
# print("\n" + "="*50)
# print(f"✅ HYBRID BATCH COMPLETE! Saved to '{output_file}'")
# print("="*50)



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

def get_llm_score(student_ans, expected_ans, max_marks, question):
    prompt = f"""
    You are a Fair University Examiner. Grade this Student Answer.
    
    GRADING RESOURCES:
    - Question: "{question}"
    - Max Marks: {max_marks}
    - Reference Answer (Guide): "{expected_ans}"
    - Student Answer: "{student_ans}"

    GRADING RUBRIC:
    1. **ASSESS UNDERSTANDING**: Does the student understand the CORE CONCEPT asked in the question? If yes, be generous.
    2. **MAIN POINTS CHECK**: Look for the key ideas in the Reference Answer. If the student includes these ideas (even with different wording), give full marks.
    3. **PARTIAL CREDIT**: If the student writes a valid answer but misses a few minor details, DEDUCT marks proportionally (e.g., -5 marks). Do NOT give 0 just for being brief.
    4. **REFERENCE IS A GUIDE**: Do not penalize the student for having a different writing style than the Reference.
    
    SAFETY CHECK (The "Poison Pill"):
    - IF the answer gives DANGEROUS advice or promotes anti-patterns, SCORE = 0.
    
    Return JSON only: {{"score": <absolute_number_out_of_{max_marks}>}}
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json", "temperature": 0.0}
        )
        # Cleanup potential Markdown formatting
        clean_text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_text)
        return float(data['score'])
    except Exception as e:
        print(f" [LLM Error: {e}] ", end="")
        return -1 # Error flag

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
    
    try: max_marks = float(row['max_marks'])
    except: max_marks = 10.0
    
    print(f"   Processing {qid}...", end="")

    # --- STEP A: VECTOR MATH & CONTEXT RETRIEVAL ---
    v_student = get_embedding(student_text).reshape(1, -1)
    v_essay   = get_embedding(str(row['expected_essay'])).reshape(1, -1)
    v_visual  = get_embedding(str(row['expected_visual'])).reshape(1, -1)
    
    # Base Similarity
    sim_essay = cosine_similarity(v_student, v_essay)[0][0]
    sim_visual = cosine_similarity(v_student, v_visual)[0][0]
    
    # Context Retrieval (Capture TEXT for LLM)
    sim_context = 0
    
    if static_mem:
        try:
            rag_res = static_mem.query(query_texts=[student_text], n_results=1)
            if rag_res['documents'] and rag_res['documents'][0]:
                v_ctx = get_embedding(context_text_found).reshape(1, -1)
                sim_context = cosine_similarity(v_student, v_ctx)[0][0]
        except: pass

    # Math Score
    math_ratio = (max(sim_essay, sim_visual) * 0.7) + (sim_context * 0.3)
    math_score = math_ratio * max_marks

    # --- STEP B: LLM JUDGE SCORE ---
    llm_score = get_llm_score(student_text, str(row['expected_essay']), max_marks, q_text)
    
    # --- STEP C: THE GATED EVALUATION FORMULA (Best Logic) ---
    
    fail_threshold = 0.25 * max_marks      # Below this = Trash
    excellence_threshold = 0.9 * max_marks # Above this = Perfect
    
    # 1. API Failure Fallback
    if llm_score == -1:
        print(" [API Error -> Using Math] ", end="")
        hybrid_score = math_score
        
    # 2. The "Hallucination Trap" (Fixes Rainbow Buttons)
    # If LLM says it's garbage (e.g., bad advice), we force the score down.
    elif llm_score < fail_threshold:
        hybrid_score = llm_score
        # print(" [⛔ Trap Triggered] ", end="")

    # 3. The "Brief Genius" Bonus (Fixes Short GOMS Answers)
    # If LLM loves it but Math hates it (because it's short), trust the LLM.
    elif llm_score > excellence_threshold and math_score < (0.6 * max_marks):
        hybrid_score = llm_score
        # print(" [✅ Brevity Bonus] ", end="")

    # 4. The Standard Blend (Balanced Approach)
    else:
        # 60% LLM (Intelligence) + 40% Math (Consistency)
        hybrid_score = (llm_score * 0.6) + (math_score * 0.4)

    # Final Clipping
    final_mark = max(0, min(hybrid_score, max_marks))

    final_results.append({
        "question_id": qid,
        "human_score": row['teacher_score'],
        "ai_score": round(final_mark, 2),
        "max_marks": max_marks,
        "method": "Gated Hybrid (Context-Aware)"
    })
    
    time.sleep(1.0)
    print(" Done.")

# --- 6. SAVE ---
output_file = "hybrid_results_gated.csv"
pd.DataFrame(final_results).to_csv(output_file, index=False)
print("\n" + "="*50)
print(f"✅ BATCH COMPLETE! Saved to '{output_file}'")
print("="*50)