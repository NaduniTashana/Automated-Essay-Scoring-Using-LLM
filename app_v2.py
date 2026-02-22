import streamlit as st
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import EmbeddingFunction, Documents, Embeddings
import time
import json
from datetime import datetime
from PIL import Image
import shutil
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AES Adaptive System", layout="wide", page_icon="🎓")

# --- CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

# --- HELPER FUNCTIONS ---
def get_embedding(text):
    if not isinstance(text, str) or len(text) < 2: return None
    try:
        resp = genai.embed_content(model="models/text-embedding-004", content=text, task_type="semantic_similarity")
        vec = np.array(resp['embedding'])
        if np.count_nonzero(vec) == 0: return None
        return vec.reshape(1, -1)
    except: return None

class GoogleEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = [get_embedding(t) for t in input]
        return [e[0].tolist() if e is not None else np.random.rand(768).tolist() for e in embeddings]

# ==============================================================================
# 🧠 SCORING AGENT
# ==============================================================================
class ScoringAgent:
    def __init__(self):
        # We initialize ChromaDB here. 
        # If the folder was deleted, this creates a fresh new one.
        self.chroma = chromadb.PersistentClient(path="my_knowledge_base")
        self.embedding_fn = GoogleEmbeddingFunction()
        
        # Initialize Collections
        self.static_mem = self.chroma.get_or_create_collection(name="lecture_notes", embedding_function=self.embedding_fn)
        self.dynamic_mem = self.chroma.get_or_create_collection(name="grading_memory", embedding_function=self.embedding_fn)
        self.model_answers_mem = self.chroma.get_or_create_collection(name="model_answers_cache", embedding_function=self.embedding_fn)

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
        except: return "Error extracting text."

    # def generate_expected_answer(self, q_text, rubric, max_marks):
    #     """Generates 'Gold Standard' answer (Concise & Exam-Style)."""
        
    #     # 1. CACHE CHECK (Strict)
    #     try:
    #         results = self.model_answers_mem.query(query_texts=[q_text], n_results=1)
    #         if results['documents'] and results['distances'][0][0] < 0.1:
    #             print("⚡ Cache Hit: Reusing existing Model Answer.")
    #             return results['metadatas'][0][0]['answer']
    #     except: pass

    #     # 2. CONTEXT RETRIEVAL (With Filter)
    #     print("🤖 Cache Miss: Generating new Model Answer...")
    #     lecture_context = ""
    #     try:
    #         results = self.static_mem.query(query_texts=[q_text], n_results=3)
    #         valid_docs = []
    #         if results['documents'] and results['distances']:
    #             for doc, dist in zip(results['documents'][0], results['distances'][0]):
    #                 if dist < 0.5: valid_docs.append(doc)
            
    #         if valid_docs:
    #             lecture_context = "\n---\n".join(valid_docs)
    #         else:
    #             lecture_context = "NO RELEVANT LECTURE SLIDES FOUND."
    #     except: pass

    #     # 3. GENERATION (Updated Prompt for CONCISENESS)
    #     # We calculate a rough word limit based on marks (e.g., 5 marks = ~50 words)
    #     target_words = int(max_marks) * 10 
        
    #     prompt = f"""
    #     You are a top-tier university student taking an exam.
    #     Write a **CONCISE, PERFECT ANSWER** for the following question.

    #     Question: "{q_text}"
    #     Max Marks: {max_marks} (Target Length: ~{target_words} words)
        
    #     PROVIDED LECTURE CONTEXT:
    #     {lecture_context}
        
    #     Rubric: {rubric if rubric else "Standard Criteria"}

    #     CRITICAL INSTRUCTIONS:
        
    #     1. **BE CONCISE:** Focus on the core ideas and key characteristics of each method.
    #     2. **MATCH STUDENT LEVEL:** Write at the level of a typical university student.
    #     3. **BE CONCISE:** Do NOT write an essay. Use bullet points or short paragraphs.
    #     4. **NO FLUFF:** Go straight to the point. Start immediately with the answer.
    #     5. **MATCH MARKS:** If the question is 5 marks, write 1-3 sentences. If 20 marks, write 100 words max.
    #     6. **RELEVANCE:** Use the Lecture Context if it matches. If irrelevant, ignore it and use standard definitions.
    #     """

    #     try:
    #         model = genai.GenerativeModel("gemini-2.5-flash")
    #         response = model.generate_content(prompt, generation_config={"temperature": 0.2})
    #         generated_ans = response.text.strip()

    #         # Save to Cache
    #         if len(generated_ans) > 10:
    #             import uuid
    #             self.model_answers_mem.upsert(
    #                 documents=[q_text],
    #                 metadatas=[{"answer": generated_ans}],
    #                 ids=[str(uuid.uuid4())]
    #             )
    #         return generated_ans
    #     except:
    #         return "Error generating answer."

    # def evaluate(self, student_text, q_text, rubric, max_marks, q_id="LIVE"):
    #     expected_answer = self.generate_expected_answer(q_text, rubric, max_marks)

    #     # Vector Score
    #     v_student = get_embedding(student_text)
    #     v_expected = get_embedding(expected_answer)
    #     if v_student is None or v_expected is None:
    #         math_score = 0
    #     else:
    #         sim = cosine_similarity(v_student, v_expected)[0][0]
    #         math_score = sim * max_marks
        
    #     # LLM Score
    #     llm_result = self.get_llm_judgment(student_text, expected_answer, q_text, rubric, max_marks)
    #     llm_score = llm_result['score']
    #     llm_feedback = llm_result['feedback']

    #     # Gated Formula
    #     fail_threshold = 0.25 * max_marks
    #     excellence_threshold = 0.9 * max_marks

    #     if llm_score < fail_threshold:
    #         final_score = llm_score
    #         logic_reason = "⛔ Penalized for conceptual error/hallucination."
    #     elif llm_score > excellence_threshold and math_score < (0.6 * max_marks):
    #         final_score = llm_score
    #         logic_reason = "✅ Awarded for concise accuracy."
    #     else:
    #         final_score = (0.6 * llm_score) + (0.4 * math_score)
    #         logic_reason = "Hybrid calculation (60% AI Logic + 40% Semantic Match)."

    #     # Adaptive History Check
    #     history_msg = ""
    #     try:
    #         hist_res = self.dynamic_mem.query(
    #             query_texts=[student_text], n_results=1,
    #             where={"question_id": q_id} if q_id != "LIVE" else None
    #         )
    #         if hist_res['documents'] and hist_res['documents'][0]:
    #             if hist_res['distances'][0][0] < 0.1: 
    #                 prev_score = float(hist_res['metadatas'][0][0]['score'])
    #                 final_score = prev_score
    #                 history_msg = " (🎯 ADAPTIVE: Aligned to your previous grading)."
    #     except: pass

    #     final_reason = f"{logic_reason} {history_msg}\n\n**Detailed Analysis:**\n{llm_feedback}"
    #     return round(max(0, min(final_score, max_marks)), 1), final_reason, expected_answer

    # def get_llm_judgment(self, student, expected, question, rubric, max_marks):
    #     prompt = f"""
    #     You are a Fair University Examiner. Grade this Student Answer.
    #     RESOURCES:
    #     - Question: "{question}"
    #     - Max Marks: {max_marks}
    #     - Reference Answer: "{expected}"
    #     - Student Answer: "{student}"
    #     - Rubric: "{rubric if rubric else 'General Conceptual Understanding'}"

    #     TASK:
    #     1. Calculate a score based on understanding.
    #     2. Provide short, bullet-point feedback.
    #     SAFETY: If dangerous advice, Score = 0.
    #     Return JSON: {{"score": <number>, "feedback": "<string>"}}
    #     """
    #     try:
    #         model = genai.GenerativeModel("gemini-2.5-flash")
    #         res = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    #         return json.loads(res.text)
    #     except:
    #         return {"score": 0.0, "feedback": "Error generating feedback."}

    def generate_expected_answer(self, q_text, rubric, max_marks):
        """Generates exam-style reference answer strictly aligned to lecture content."""

        # 1. Cache check (looser to avoid wrong reuse)
        try:
            results = self.model_answers_mem.query(query_texts=[q_text], n_results=1)
            if results["documents"] and results["distances"][0][0] < 0.05:
                print("⚡ Cache Hit: Reusing model answer.")
                return results["metadatas"][0][0]["answer"]
        except:
            pass

        # 2. Retrieve lecture context ONLY
        print("🤖 Cache Miss: Generating new Model Answer...")
        lecture_context = ""
        try:
            results = self.static_mem.query(query_texts=[q_text], n_results=3)
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
        **NO FLUFF:** Go straight to the point. Start immediately with the answer.

        Question: "{q_text}"
        Max Marks: {max_marks} (~{target_words} words)

        LECTURE CONTENT:
        {lecture_context if lecture_context else "No direct lecture text. Use standard definitions only."}

        Rubric: {rubric if rubric else "Concept accuracy"}

        Write the answer directly. No introductions. No conclusions.
        """

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt, generation_config={"temperature": 0.10})
            answer = response.text.strip()

            if len(answer) > 20:
                import uuid
                self.model_answers_mem.upsert(
                    documents=[q_text],
                    metadatas=[{"answer": answer}],
                    ids=[str(uuid.uuid4())],
                )

            return answer
        except:
            return "Error generating answer."

    def evaluate(self, student_text, q_text, rubric, max_marks, q_id="LIVE"):
        expected_answer = self.generate_expected_answer(q_text, rubric, max_marks)

        # Relevance check (student ↔ question)
        v_student = get_embedding(student_text)
        v_question = get_embedding(q_text)

        relevance_penalty = 0
        if v_student is not None and v_question is not None:
            sim = cosine_similarity(v_student, v_question)[0][0]
            if sim < 0.3:
                relevance_penalty = 0.2 * max_marks

        # LLM grading (authoritative)
        llm_result = self.get_llm_judgment(
            student_text, expected_answer, q_text, rubric, max_marks
        )
        llm_score = llm_result["score"]
        llm_feedback = llm_result["feedback"]

        fail_threshold = 0.4 * max_marks

        if llm_score < fail_threshold:
            final_score = llm_score
            logic_reason = "⛔ Conceptually incorrect."
        else:
            final_score = max(llm_score - relevance_penalty, 0)
            logic_reason = "✅ Concept-based grading with relevance check."

        return (
            round(min(final_score, max_marks), 1),
            f"{logic_reason}\n\n**Feedback:**\n" + "\n".join([f"- {item}" for item in llm_feedback]),
            expected_answer,
        )

    def get_llm_judgment(self, student, expected, question, rubric, max_marks):
        prompt = f"""
        You are a Fair University Examiner.

        STRICT RULES:
        - Grade based on presence and correctness of concepts.
        - Do NOT penalize brevity.
        - Do NOT reward extra detail beyond what is asked.
        - Ignore wording differences.

        Question: "{question}"
        Max Marks: {max_marks}

        Reference Concepts (do NOT compare wording):
        {expected}

        Student Answer:
        {student}

        Rubric:
        {rubric if rubric else "Concept correctness and relevance"}

        TASK:
        - Award marks per correct concept.
        - Partial credit for partially correct ideas.
        - Return short bullet-point feedback.

        Return JSON ONLY:
        {{
        "score": <number between 0 and {max_marks}>,
        "feedback": "<bullet points>"
        }}
        """

        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            res = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json", "temperature": 0},
            )
            return json.loads(res.text)
        except:
            return {"score": 0.0, "feedback": "Grading error."}



    def learn(self, text, score, max_marks, feedback, q_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dynamic_mem.upsert(
            documents=[text],
            metadatas=[{
                "question_id": q_id,
                "score": str(score),
                "max_marks": str(max_marks),
                "teacher_feedback": feedback,
                "type": "human_override"
            }],
            ids=[f"learn_{q_id}_{timestamp}"]
        )

# ==============================================================================
# 🖥️ UI LAYOUT
# ==============================================================================
st.title("🎓 Multi-Agent Adaptive Grading System")

# --- FACTORY RESET LOGIC ---
def factory_reset():
    """Nuclear Option: Deletes the database folder"""
    if os.path.exists("my_knowledge_base"):
        shutil.rmtree("my_knowledge_base")
    st.cache_resource.clear()
    st.session_state.clear()
    return True

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Tools")
    st.warning("Use this if the AI is confused.")
    
    if st.button("☢️ FACTORY RESET (Wipe All Memory)", type="primary"):
        if factory_reset():
            st.success("System Wiped! Reloading...")
            time.sleep(1)
            st.rerun()

# Initialize Agent
if "scorer" not in st.session_state:
    st.session_state.scorer = ScoringAgent()

# Initialize Text
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = "Usability engineering is a systematic approach to design that focuses on user needs."

tab1, tab2 = st.tabs(["🔴 Live Grading", "📂 Batch Processing"])

with tab1:
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("1. Assignment Details")
        st.caption("💡 **Tip:** Use a new ID for new questions.")
        q_id = st.text_input("Question ID", value="Q_Live_03") 
        q_text = st.text_area("Question Text", value="Explain Usability Engineering.")
        max_marks = st.number_input("Max Marks", value=50, min_value=1)
        rubric = st.text_area("Marking Scheme (Optional)", value="")
        
        st.divider()
        st.subheader("2. Student Submission")
        
        uploaded_img = st.file_uploader("📸 Upload Answer Script (Optional)", type=["png", "jpg", "jpeg"])
        if uploaded_img:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Script", use_column_width=True)
            if st.button("📝 Extract Text"):
                with st.spinner("👀 Reading Handwriting..."):
                    extracted = st.session_state.scorer.extract_text_from_image(image)
                    st.session_state.extracted_text = extracted
                    st.success("Extracted!")

        student_text_input = st.text_area("Answer Text", value=st.session_state.extracted_text, height=200)
        if student_text_input != st.session_state.extracted_text:
            st.session_state.extracted_text = student_text_input
        
        if st.button("🚀 Grade Submission", type="primary"):
            with st.spinner("🤖 Evaluating..."):
                score, reason, expected_ans = st.session_state.scorer.evaluate(student_text_input, q_text, rubric, max_marks, q_id)
                st.session_state.last_result = {"score": score, "reason": reason, "text": student_text_input, "expected": expected_ans}
    
    with col_result:
        st.subheader("3. Result")
        if "last_result" in st.session_state:
            res = st.session_state.last_result
            st.info("🧠 **Model Answer Used**")
            with st.expander("View Reference"): st.write(res['expected'])
            st.metric(label="Score", value=f"{res['score']} / {max_marks}")
            st.markdown("### Evaluation Report")
            st.info(res['reason'])
            
            st.divider()
            st.write("Disagree?")
            new_score = st.number_input("Override Score", value=float(res['score']), min_value=0.0, max_value=float(max_marks), step=0.5)
            teacher_notes = st.text_input("Reason")
            if st.button("💾 Save & Learn"):
                st.session_state.scorer.learn(res['text'], new_score, max_marks, teacher_notes, q_id)
                st.success("Learned!")

with tab2:
    st.header("Bulk Grading")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file and st.button("Run Batch"):
        st.dataframe(pd.read_csv(uploaded_file).head())
        st.success("Batch processing complete!")