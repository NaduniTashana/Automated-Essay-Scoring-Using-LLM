import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import EmbeddingFunction, Documents, Embeddings
import time

# --- 1. CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

# WEIGHTS (Must match your Math Script)
W_SEMANTIC = 0.5
W_RUBRIC   = 0.3
W_CONTEXT  = 0.2

# --- 2. HELPER: EMBEDDING FUNCTION ---
def get_embedding(text):
    if not isinstance(text, str) or len(text) < 2: return np.zeros((1, 768))
    try:
        resp = genai.embed_content(model="models/text-embedding-004", content=text, task_type="semantic_similarity")
        return np.array(resp['embedding']).reshape(1, -1)
    except: return np.zeros((1, 768))

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [get_embedding(t)[0].tolist() for t in input]

# ==============================================================================
# 🤖 AGENT 1: THE SCORING AGENT (The Expert)
# Responsibility: Calculates Score + Generates "Why" Feedback
# ==============================================================================
class ScoringAgent:
    def __init__(self):
        print("   🧠 [Scoring Agent] Initializing Knowledge Bases...")
        self.chroma = chromadb.PersistentClient(path="my_knowledge_base")
        
        # Connect to Brains
        try:
            self.static_mem = self.chroma.get_collection(name="lecture_notes", embedding_function=GoogleEmbeddingFunction())
            self.dynamic_mem = self.chroma.get_collection(name="grading_memory", embedding_function=GoogleEmbeddingFunction())
        except:
            print("   ⚠️ [Scoring Agent] Warning: Memories not found. Running in Logic-Only mode.")
            self.static_mem = None
            self.dynamic_mem = None

    def evaluate(self, student_text, question_data):
        """
        The Core Intelligence. Returns a dictionary with Score AND Feedback.
        """
        # 1. Vectorize Everything
        v_student = get_embedding(student_text)
        v_essay   = get_embedding(str(question_data['expected_essay']))
        v_visual  = get_embedding(str(question_data['expected_visual']))
        v_rubric  = get_embedding(str(question_data['marking_scheme']))
        
        # 2. Similarity Calculations
        sim_essay = cosine_similarity(v_student, v_essay)[0][0]
        sim_visual = cosine_similarity(v_student, v_visual)[0][0]
        sim_rubric = cosine_similarity(v_student, v_rubric)[0][0]
        
        # Determine Style (Did they draw or write?)
        style = "Visual/Diagrammatic" if sim_visual > sim_essay else "Textual/Theoretical"
        best_content_sim = max(sim_essay, sim_visual)

        # 3. Context Check (RAG)
        sim_context = 0
        if self.static_mem:
            try:
                rag_res = self.static_mem.query(query_texts=[student_text], n_results=1)
                if rag_res['documents'][0]:
                    v_ctx = get_embedding(rag_res['documents'][0][0])
                    sim_context = cosine_similarity(v_student, v_ctx)[0][0]
            except: pass

        # 4. History Check (Adaptive)
        history_msg = ""
        past_score_ratio = 0
        history_applied = False
        
        if self.dynamic_mem:
            try:
                hist_res = self.dynamic_mem.query(
                    query_texts=[student_text], n_results=1, 
                    where={"question_id": question_data['question_id']}
                )
                if hist_res['documents'][0]:
                    v_past = get_embedding(hist_res['documents'][0][0])
                    sim_hist = cosine_similarity(v_student, v_past)[0][0]
                    
                    if sim_hist > 0.85: # Threshold
                        history_applied = True
                        past_score = float(hist_res['metadatas'][0][0]['score'])
                        past_score_ratio = past_score / float(question_data['max_marks'])
                        history_msg = " (Adjusted based on similarity to a past graded script)."
            except: pass

        # 5. Final Score Calculation
        raw_ratio = (best_content_sim * W_SEMANTIC) + (sim_rubric * W_RUBRIC) + (sim_context * W_CONTEXT)
        
        if history_applied:
            final_ratio = (0.6 * raw_ratio) + (0.4 * past_score_ratio)
        else:
            final_ratio = raw_ratio
            
        final_score = round(max(0, min(final_ratio * float(question_data['max_marks']), float(question_data['max_marks']))), 1)

        # 6. GENERATE FEEDBACK (The "Why")
        feedback = self._construct_feedback_sentence(style, best_content_sim, sim_rubric, sim_context, history_msg)
        
        return {
            "score": final_score,
            "max_marks": question_data['max_marks'],
            "style": style,
            "feedback": feedback
        }

    def _construct_feedback_sentence(self, style, sim_content, sim_rubric, sim_context, hist_msg):
        """
        Translates numbers into English.
        """
        comments = []
        
        # Comment on Strength
        if sim_content > 0.75:
            comments.append(f"Excellent {style} explanation of the core concept.")
        elif sim_content > 0.5:
            comments.append(f"Good effort on the {style} description, but lacks some depth.")
        else:
            comments.append(f"The {style} explanation was vague or off-topic.")

        # Comment on Rubric/Keywords
        if sim_rubric < 0.6:
            comments.append("Key technical terms from the marking scheme were missing.")
        
        # Comment on Context (Lecture Notes)
        if sim_context > 0.6:
            comments.append("Demonstrated strong alignment with lecture definitions.")
        
        # Join and add history note
        full_text = " ".join(comments) + hist_msg
        return full_text

# ==============================================================================
# 🌟 AGENT 2: THE PRIME AGENT (The Manager)
# Responsibility: UI, Work Delegation, Human Validation
# ==============================================================================
class PrimeAgent:
    def __init__(self):
        print("🌟 [Prime Agent] System Online. Connecting to Scoring Agent...")
        self.scorer = ScoringAgent()
        
        # Load the "Gold Standard" knowledge
        print("🌟 [Prime Agent] Loading Exam Standards...")
        self.standards = pd.read_csv("expected_answers_dual.csv")

    def process_live_batch(self, csv_file):
        """
        Simulates the live grading of students.
        """
        # Load inputs
        df_students = pd.read_csv(csv_file)
        
        # Clean merge (same logic as your math script to avoid errors)
        cols_to_drop = [c for c in df_students.columns if c in self.standards.columns and c != 'question_id']
        df_students = df_students.drop(columns=cols_to_drop)
        
        # Merge to get the Question Data
        df_merged = pd.merge(df_students, self.standards, on='question_id', how='left')
        
        print(f"\n🚀 STARTING LIVE GRADING SESSION ({len(df_merged)} Submissions)...\n")
        time.sleep(1)

        results = []

        for index, row in df_merged.iterrows():
            student_id = row['student_id']
            q_id = row['question_id']
            text = str(row['extracted_text'])
            
            # 1. DELEGATION: Prime Agent asks Scoring Agent
            # Note: We pass the whole row (series) as the 'question_data'
            grading_result = self.scorer.evaluate(text, row)
            
            # 2. PRESENTATION: Prime Agent displays result
            print(f"📄 Student {student_id} | Q: {q_id}")
            print(f"   Detected Style: {grading_result['style']}")
            print(f"   🤖 AI Score:   {grading_result['score']} / {grading_result['max_marks']}")
            print(f"   📝 Feedback:   {grading_result['feedback']}")
            print("-" * 50)
            
            # Store for saving
            results.append({
                "student_id": student_id,
                "question_id": q_id,
                "ai_score": grading_result['score'],
                "feedback": grading_result['feedback']
            })
            
            # Small pause to look realistic
            # time.sleep(0.1)

        return pd.DataFrame(results)

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    system = PrimeAgent()
    final_output = system.process_live_batch("cleaned_dataset.csv")
    
    # Save the feedback report
    final_output.to_csv("final_student_feedback.csv", index=False)
    print("\n✅ Grading Complete. Feedback sent to 'final_student_feedback.csv'.")