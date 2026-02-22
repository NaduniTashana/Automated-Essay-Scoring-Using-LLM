import pandas as pd
import google.generativeai as genai
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
import os
import time

# --- 1. CONFIGURATION ---
# Replace with your actual API Key
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"  
genai.configure(api_key=GENAI_API_KEY)

# Use the fast and smart Gemini 2.5 Flash model
try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("Model Selected: gemini-2.5-flash")
except:
    print("2.5 Flash not found, falling back to Flash-Lite")
    model = genai.GenerativeModel("gemini-2.5-flash-lite")

# --- 2. RAG MEMORY SETUP ---
print("Connecting to Static Memory (Lecture Notes)...")
chroma_client = chromadb.PersistentClient(path="my_knowledge_base")

# Define the Embedding Function (Must match the one used to build the DB)
class GoogleGenerativeAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_query" # 'query' type is best for searching
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                # Return a zero vector if embedding fails (safety fallback)
                embeddings.append([0]*768) 
        return embeddings

try:
    collection = chroma_client.get_collection(
        name="lecture_notes",
        embedding_function=GoogleGenerativeAIEmbeddingFunction()
    )
    print("Successfully connected to RAG Database.")
except Exception as e:
    print(f"Error: Could not find RAG database. Did you run 'rag_engine.py'? \n{e}")
    exit()

# --- 3. LOAD QUESTIONS ---
input_csv = "questions_db.csv"
if not os.path.exists(input_csv):
    print(f"Error: '{input_csv}' not found.")
    print("   Please ensure questions_db.csv is in the folder.")
    exit()

df = pd.read_csv(input_csv)
print(f"--- Generating Expected Answers for {len(df)} Questions ---")

golden_answers = []

# --- 4. GENERATION LOOP ---
for index, row in df.iterrows():
    qid = row['question_id']
    q_text = row['question_text']
    marking = row['marking_scheme']
    max_marks = str(row['max_marks']) # Convert to string for the prompt
    
    print(f"\n Processing {qid} ({max_marks} Marks)...", end="")
    
    # --- STEP A: RETRIEVE CONTEXT FROM LECTURES ---
    try:
        results = collection.query(
            query_texts=[q_text],
            n_results=3 # Get top 3 most relevant pages
        )
        if results['documents']:
            lecture_context = "\n\n".join(results['documents'][0])
        else:
            lecture_context = "No specific lecture notes found."
    except Exception as e:
        print(f" (RAG Error: {e})", end="")
        lecture_context = "No context available."

    # --- STEP B: GENERATE VERSION 1 (PURE ESSAY) ---
    # Prompt for students who write text-only answers
    prompt_essay = f"""
    You are an expert university examiner. Write a STANDARD MODEL ANSWER for:
    
    Question: "{q_text}"
    Max Marks Available: {max_marks}
    
    SOURCE MATERIAL (Lecture Notes):
    {lecture_context}
    
    MARKING SCHEME:
    {marking}
    
    CRITICAL INSTRUCTION ON LENGTH & DEPTH:
    This question is worth {max_marks} marks. 
    - Adjust the length and detail of your answer to match this score perfectly.
    - If the marks are LOW: Be concise and direct.
    - If the marks are HIGH (e.g., >20): You MUST provide deep theoretical context, examples, and critical analysis.
    - Ensure you cover enough points to justify awarding the full {max_marks} marks.
    
    RESTRICTION:
    Do NOT describe any diagrams, sketches, or visual elements in this version. Use text only.
    """

    # --- STEP C: GENERATE VERSION 2 (VISUAL DESCRIPTION) ---
    # Prompt for students who draw diagrams/sketches
    prompt_visual = f"""
    You are an expert university examiner. Write a DESCRIPTIVE MODEL ANSWER for:
    
    Question: "{q_text}"
    Max Marks Available: {max_marks}
    
    SOURCE MATERIAL:
    {lecture_context}
    
    MARKING SCHEME:
    {marking}
    
    INSTRUCTION:
    1. Write a theoretical answer scaled to justify {max_marks} marks.
    2. Crucially, include a detailed description of a DIAGRAM, INTERFACE SKETCH, or CHART that a student might draw to explain this concept.
    
    FORMAT FOR DIAGRAMS:
    Use double brackets like this: 
    "[[Diagram: A [shape] representing [concept]... containing labels like 'X' and 'Y'...]]"
    
    Use visual vocabulary: "box", "arrow", "grid", "dropdown", "button", "layout".
    Combine the text explanation with this diagram description.
    """

    try:
        # Generate both versions
        resp_essay = model.generate_content(prompt_essay).text.strip()
        resp_visual = model.generate_content(prompt_visual).text.strip()
        
        golden_answers.append({
            "question_id": qid,
            "question_text": q_text,
            "max_marks": max_marks,
            "marking_scheme": marking,
            "expected_essay": resp_essay,   # Version A (Text)
            "expected_visual": resp_visual  # Version B (Text + Sketch)
        })
        print(" Done.")
        time.sleep(1.5) # Polite pause to avoid hitting rate limits
        
    except Exception as e:
        print(f" API Error: {e}")

# --- 5. SAVE RESULTS ---
output_file = "expected_answers_dual.csv"
output_df = pd.DataFrame(golden_answers)
output_df.to_csv(output_file, index=False)

print("\n" + "="*50)
print("GENERATION COMPLETE!")
print(f"   Saved to: '{output_file}'")
print("   - Column 'expected_essay': For text-only grading")
print("   - Column 'expected_visual': For diagram-based grading")
print("="*50)