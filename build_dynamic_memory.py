import pandas as pd
import chromadb
import google.generativeai as genai
from chromadb import EmbeddingFunction, Documents, Embeddings
import os

# --- CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"
genai.configure(api_key=GENAI_API_KEY)

# --- GOOGLE EMBEDDING FUNCTION ---
class GoogleGenerativeAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                    title="Student Answer"
                )
                embeddings.append(response["embedding"])
            except:
                embeddings.append([0]*768)
        return embeddings

# --- CONNECT TO DATABASE ---
chroma_client = chromadb.PersistentClient(path="my_knowledge_base")
embedding_function = GoogleGenerativeAIEmbeddingFunction()

# Create a NEW collection for Experience
print("Creating Dynamic Grading Memory...")
collection = chroma_client.get_or_create_collection(
    name="grading_memory",
    embedding_function=embedding_function
)

# --- LOAD INITIAL DATA (OPTIONAL) ---
# If you have your 'initial_dataset_multipage.csv' with teacher scores, 
# we can seed the memory with it.
if os.path.exists("cleaned_dataset.csv"):
    df = pd.read_csv("cleaned_dataset.csv")
    print(f"   Seeding memory with {len(df)} past answers...")
    
    documents = []
    ids = []
    metadatas = []
    
    for index, row in df.iterrows():
        # Only add if there is a valid score
        if pd.notna(row.get('teacher_score')):
            documents.append(str(row['extracted_text']))
            ids.append(f"past_{row['question_id']}_{row['student_id']}")
            metadatas.append({
                "question_id": str(row['question_id']),
                "score": str(row['teacher_score']),
                "type": "initial_seed"
            })
    
    # Save to DB
    if documents:
        batch_size = 20
        for i in range(0, len(documents), batch_size):
            collection.upsert(
                documents=documents[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            print(f"   Batch {i} saved.")

print("Dynamic Memory Ready!")