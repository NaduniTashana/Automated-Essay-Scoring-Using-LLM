# import os
# import re
# import pandas as pd
# import google.generativeai as genai
# import PIL.Image

# # --- 1. CONFIGURATION ---
# GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"  # Paste your key here
# genai.configure(api_key=GENAI_API_KEY)
# model = genai.GenerativeModel("gemini-2.5-flash")

# root_dir = "images"  # Your main folder
# dataset_rows = []

# # Load questions
# if os.path.exists("questions_db.csv"):
#     df_questions = pd.read_csv("questions_db.csv")
#     question_map = df_questions.set_index("question_id").to_dict("index")
# else:
#     print("❌ Error: questions_db.csv not found!")
#     exit()

# print("--- Starting Multi-Page Extraction ---")

# # --- 2. LOOP THROUGH QUESTION FOLDERS ---
# for q_folder in sorted(os.listdir(root_dir)):
#     q_path = os.path.join(root_dir, q_folder)
    
#     if os.path.isdir(q_path):
#         qid = q_folder  # e.g., "Q1"
#         print(f"\n📂 Entering Folder: {qid}")

#         if qid not in question_map:
#             print(f"   ⚠️ Skipping {qid} (Not in questions_db.csv)")
#             continue

#         q_text = question_map[qid]["question_text"]
#         q_scheme = question_map[qid]["marking_scheme"]

#         # --- 3. GROUP FILES BY STUDENT ID ---
#         # We want to turn ['student_01.jpg', 'student_03_p1.jpg', 'student_03_p2.jpg']
#         # Into -> {'student_01': ['student_01.jpg'], 'student_03': ['student_03_p1.jpg', 'student_03_p2.jpg']}
        
#         student_files = {}
        
#         for file in sorted(os.listdir(q_path)):
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 # Logic: Remove _p1, _p2 to find the 'base' student ID
#                 # If name is 'student_03_p1.jpg', base_id is 'student_03'
#                 base_id = re.sub(r'_p\d+', '', os.path.splitext(file)[0])
                
#                 if base_id not in student_files:
#                     student_files[base_id] = []
#                 student_files[base_id].append(file)

#         # --- 4. PROCESS EACH STUDENT (BATCH PAGES) ---
#         for student_id, files in student_files.items():
#             print(f"   Processing Student: {student_id} ({len(files)} pages)...")
            
#             full_text = ""
#             file_paths_record = []

#             # Sort files to ensure Page 1 comes before Page 2
#             files.sort() 

#             try:
#                 for img_file in files:
#                     img_path = os.path.join(q_path, img_file)
#                     file_paths_record.append(img_path)
                    
#                     # Extract Text for this page
#                     img = PIL.Image.open(img_path)
#                     response = model.generate_content(["Transcribe this handwritten page exactly.", img])
#                     page_text = response.text.strip()
                    
#                     full_text += f"\n{page_text}"

#                 # --- ADD SINGLE ROW TO DATASET ---
#                 dataset_rows.append({
#                     "question_id": qid,
#                     "student_id": student_id,
#                     "question": q_text,
#                     "marking_scheme": q_scheme,
#                     "image_paths": " | ".join(file_paths_record), # Stores "Q1/s3_p1.jpg | Q1/s3_p2.jpg"
#                     "extracted_text": full_text.strip(),
#                     "teacher_score": ""  # Fill this manually later
#                 })

#             except Exception as e:
#                 print(f"   ❌ Error processing {student_id}: {e}")

# # --- 5. SAVE CSV ---
# output_file = "initial_dataset_multipage.csv"
# df_final = pd.DataFrame(dataset_rows)
# df_final.to_csv(output_file, index=False)

# print("\n" + "="*50)
# print(f"✅ DONE! Created '{output_file}'")
# print(f"   - Total Unique Answers: {len(df_final)}")
# print("   - Open the file and add your scores in the 'teacher_score' column.")
# print("="*50)



# import os
# import re
# import time
# import pandas as pd
# import google.generativeai as genai
# import PIL.Image

# # --- 1. CONFIGURATION ---
# GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA" 
# genai.configure(api_key=GENAI_API_KEY)

# # CRITICAL UPDATE: Use "gemini-2.5-flash-lite"
# # "gemini-2.5-flash" has a limit of ~20/day.
# # "gemini-2.5-flash-lite" allows ~1,500/day and supports images.
# try:
#     model = genai.GenerativeModel("gemini-2.5-flash-lite")
# except:
#     # Fallback to 2.0 Flash Lite if 2.5 Lite isn't active in your region yet
#     print("⚠️ 2.5 Flash-Lite not found, trying 2.0 Flash-Lite...")
#     model = genai.GenerativeModel("gemini-2.0-flash-lite-preview-02-05")

# root_dir = "images"  
# output_file = "initial_dataset_multipage.csv"
# dataset_rows = []

# # # --- 2. STRICT PROMPT (Removes "Additional Notes") ---
# # strict_prompt = """
# # You are a data entry assistant. Transcribe ONLY the student's handwritten answer from this image.
# # RULES:
# # 1. Ignore all red ink marks, ticks, circles, and scores added by the teacher.
# # 2. Ignore any text that looks like a grade (e.g., "5", "10/10", "Good").
# # 3. Do NOT describe the image (e.g., do not say "Here is the transcription").
# # 4. Do NOT describe visual features (e.g., "The word is underlined").
# # 5. Output ONLY the raw text of the answer.
# # """

# strict_prompt = """
# You are a research assistant digitizing handwritten student answers. 

# **INSTRUCTIONS:**
# 1. **Transcribe Text:** Transcribe the student's handwritten text exactly as it appears.
# 2. **Handle Diagrams/Drawings:** If the student has drawn a diagram, chart, UI sketch, or table:
#    - Describe it strictly within double brackets like this: `[[Diagram: User Interface sketch of a calculator showing number keys and a display reading 132]]`.
#    - If text is written *inside* a diagram (e.g., inside boxes), transcribe it as part of the diagram description or as a structured list below it.
# 3. **Ignore Teacher Marks:** COMPLETELY IGNORE all red ink, ticks, checkmarks, circles, and grading scores added by the teacher.
# 4. **Format:** Output raw text only. No conversational filler like "Here is the transcription."
# """

# # --- 3. RESUME LOGIC ---
# processed_ids = set()
# if os.path.exists(output_file):
#     print(f"🔄 Found existing {output_file}. Loading progress...")
#     try:
#         df_existing = pd.read_csv(output_file)
#         dataset_rows = df_existing.to_dict('records')
#         for index, row in df_existing.iterrows():
#             unique_key = f"{row['question_id']}_{row['student_id']}"
#             processed_ids.add(unique_key)
#         print(f"✅ Already processed {len(processed_ids)} students. Resuming...")
#     except Exception as e:
#         print(f"⚠️ Could not load file: {e}. Starting fresh.")
# else:
#     print("🚀 Starting fresh...")

# if os.path.exists("questions_db.csv"):
#     df_questions = pd.read_csv("questions_db.csv")
#     question_map = df_questions.set_index("question_id").to_dict("index")
# else:
#     print("❌ Error: questions_db.csv not found!")
#     exit()

# print("--- Starting Extraction (Model: Gemini 2.5 Flash-Lite) ---")

# # --- 4. LOOP THROUGH FOLDERS ---
# for q_folder in sorted(os.listdir(root_dir)):
#     q_path = os.path.join(root_dir, q_folder)
    
#     if os.path.isdir(q_path):
#         qid = q_folder
#         if qid not in question_map:
#             continue

#         q_text = question_map[qid]["question_text"]
#         q_scheme = question_map[qid]["marking_scheme"]

#         student_files = {}
#         for file in sorted(os.listdir(q_path)):
#             if file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 base_id = re.sub(r'_p\d+', '', os.path.splitext(file)[0])
#                 if base_id not in student_files:
#                     student_files[base_id] = []
#                 student_files[base_id].append(file)

#         # --- 5. PROCESS STUDENTS ---
#         for student_id, files in student_files.items():
#             unique_key = f"{qid}_{student_id}"
#             if unique_key in processed_ids:
#                 continue

#             print(f"   Processing {qid} - {student_id} ({len(files)} pages)...")
            
#             full_text = ""
#             file_paths_record = []
#             files.sort() 

#             try:
#                 # With Flash-Lite, we can go faster (wait 2 seconds instead of 4)
#                 time.sleep(2) 

#                 for img_file in files:
#                     img_path = os.path.join(q_path, img_file)
#                     file_paths_record.append(img_path)
                    
#                     img = PIL.Image.open(img_path)
                    
#                     try:
#                         response = model.generate_content([strict_prompt, img])
#                         page_text = response.text.strip()
#                     except Exception as e:
#                         if "429" in str(e):
#                             print("      ⚠️ Rate Limit Hit. Waiting 30s...")
#                             time.sleep(30)
#                             response = model.generate_content([strict_prompt, img])
#                             page_text = response.text.strip()
#                         else:
#                             raise e 

#                     full_text += f"\n{page_text}"

#                 # Save Data
#                 new_row = {
#                     "question_id": qid,
#                     "student_id": student_id,
#                     "question": q_text,
#                     "marking_scheme": q_scheme,
#                     "image_paths": " | ".join(file_paths_record),
#                     "extracted_text": full_text.strip(),
#                     "teacher_score": ""
#                 }
#                 dataset_rows.append(new_row)
#                 pd.DataFrame(dataset_rows).to_csv(output_file, index=False)
#                 processed_ids.add(unique_key)

#             except Exception as e:
#                 print(f"   ❌ Error processing {student_id}: {e}")

# print("\n" + "="*50)
# print(f"✅ FINAL DONE! Check '{output_file}'")
# print("="*50)


#with paid API
import os
import re
import time
import pandas as pd
import google.generativeai as genai
import PIL.Image

# --- 1. CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

# --- 2. SMART MODEL SELECTOR ---
# We try these models in order until one works.
candidate_models = [
    "gemini-2.5-flash",        # Your requested model
    "gemini-2.5-flash-lite",   # The lite version
    "gemini-2.0-flash-exp",    # High-limit experimental models
]

selected_model = None

print("🔍 Auto-detecting the best available model for you...")
for m_name in candidate_models:
    try:
        print(f"   Testing: {m_name}...", end="")
        test_model = genai.GenerativeModel(m_name)
        # We send a tiny "Hello" to check if the model is alive and accessible
        test_model.generate_content("Hello")
        selected_model = test_model
        print(" ✅ WORKS!")
        break
    except Exception as e:
        print(" ❌ Failed.")

if not selected_model:
    print("\n⚠️ CRITICAL ERROR: No models are working with your Key/Region.")
    print("Please run 'pip install -U google-generativeai' to update your library.")
    exit()

print(f"🚀 Selected Model: {selected_model.model_name}")

root_dir = "images"  
output_file = "initial_dataset_multipage.csv"
dataset_rows = []

# --- 3. YOUR STRICT PROMPT ---
strict_prompt = """
You are a research assistant digitizing handwritten student answers. 

**INSTRUCTIONS:**
1. **Transcribe Text:** Transcribe the student's handwritten text exactly as it appears.
2. **Handle Diagrams/Drawings:** If the student has drawn a diagram, chart, UI sketch, or table:
   - Describe it strictly within double brackets like this: `[[Diagram: User Interface sketch of a calculator showing number keys and a display reading 132]]`.
   - If text is written *inside* a diagram (e.g., inside boxes), transcribe it as part of the diagram description or as a structured list below it.
3. **Ignore Teacher Marks:** COMPLETELY IGNORE all red ink, ticks, checkmarks, circles, and grading scores added by the teacher.
4. **Format:** Output raw text only. No conversational filler like "Here is the transcription."
"""

# --- 4. RESUME LOGIC (With Corrupt File Fix) ---
processed_ids = set()

if os.path.exists(output_file):
    print(f"🔄 Found existing {output_file}...")
    try:
        df_existing = pd.read_csv(output_file)
        if df_existing.empty:
            raise ValueError("File is empty")
        
        dataset_rows = df_existing.to_dict('records')
        for index, row in df_existing.iterrows():
            unique_key = f"{row['question_id']}_{row['student_id']}"
            processed_ids.add(unique_key)
        print(f"✅ Already processed {len(processed_ids)} students. Resuming...")
    except Exception as e:
        print(f"⚠️ File was corrupt or empty ({e}). Deleting and starting fresh.")
        os.remove(output_file) # Auto-delete the bad file
        dataset_rows = []
else:
    print("🚀 Starting fresh...")

if os.path.exists("questions_db.csv"):
    df_questions = pd.read_csv("questions_db.csv")
    question_map = df_questions.set_index("question_id").to_dict("index")
else:
    print("❌ Error: questions_db.csv not found!")
    exit()

print("--- Starting Extraction ---")

# --- 5. LOOP THROUGH FOLDERS ---
for q_folder in sorted(os.listdir(root_dir)):
    q_path = os.path.join(root_dir, q_folder)
    
    if os.path.isdir(q_path):
        qid = q_folder
        if qid not in question_map:
            continue

        q_text = question_map[qid]["question_text"]
        q_scheme = question_map[qid]["marking_scheme"]

        student_files = {}
        for file in sorted(os.listdir(q_path)):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_id = re.sub(r'_p\d+', '', os.path.splitext(file)[0])
                if base_id not in student_files:
                    student_files[base_id] = []
                student_files[base_id].append(file)

        # --- 6. PROCESS STUDENTS ---
        for student_id, files in student_files.items():
            unique_key = f"{qid}_{student_id}"
            if unique_key in processed_ids:
                continue

            print(f"   Processing {qid} - {student_id} ({len(files)} pages)...")
            
            full_text = ""
            file_paths_record = []
            files.sort() 

            try:
                # No sleep needed for paid account, but we keep a tiny safety buffer
                # just in case the model is busy.
                time.sleep(0.5)

                for img_file in files:
                    img_path = os.path.join(q_path, img_file)
                    file_paths_record.append(img_path)
                    
                    img = PIL.Image.open(img_path)
                    
                    try:
                        response = selected_model.generate_content([strict_prompt, img])
                        page_text = response.text.strip()
                    except Exception as e:
                        if "429" in str(e):
                            print("      ⚠️ Rate Limit Spike. Pausing 10s...")
                            time.sleep(10)
                            response = selected_model.generate_content([strict_prompt, img])
                            page_text = response.text.strip()
                        else:
                            raise e 

                    full_text += f"\n{page_text}"

                # Save Data
                new_row = {
                    "question_id": qid,
                    "student_id": student_id,
                    "question": q_text,
                    "marking_scheme": q_scheme,
                    "image_paths": " | ".join(file_paths_record),
                    "extracted_text": full_text.strip(),
                    "teacher_score": ""
                }
                dataset_rows.append(new_row)
                pd.DataFrame(dataset_rows).to_csv(output_file, index=False)
                processed_ids.add(unique_key)

            except Exception as e:
                print(f"   ❌ Error processing {student_id}: {e}")

print("\n" + "="*50)
print(f"✅ FINAL DONE! Check '{output_file}'")
print("="*50)