import os
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
import time

# --- 1. CONFIGURATION ---
GENAI_API_KEY = "AIzaSyDMjmQn450aqp9B1n0DMRNfXMtF0QiXedA"  
genai.configure(api_key=GENAI_API_KEY)

# --- 2. EMBEDDING FUNCTION (The "Google" Brain) ---
class GoogleGenerativeAIEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        # We process one by one to handle errors safely
        for text in input:
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                    title="Lecture Notes"
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"      ⚠️ Embedding Error: {e}")
                # Return a blank vector if it fails (rare)
                embeddings.append([0]*768) 
        return embeddings

# --- 3. DATABASE SETUP ---
print("Connecting to Database...")
chroma_client = chromadb.PersistentClient(path="my_knowledge_base")
embedding_function = GoogleGenerativeAIEmbeddingFunction()

# Create collection
collection = chroma_client.get_or_create_collection(
    name="lecture_notes",
    embedding_function=embedding_function
)

def build_knowledge_base():
    materials_dir = "materials"
    
    # Check folders
    if not os.path.exists(materials_dir):
        print(f"Error: Folder '{materials_dir}' not found.")
        print("   Please create a folder named 'materials' and put your PDFs inside.")
        return

    # Check files
    files = [f for f in os.listdir(materials_dir) if f.lower().endswith(".pdf")]
    if not files:
        print("No PDF files found inside 'materials' folder!")
        return

    print(f"--- Found {len(files)} PDFs. Starting Build... ---")

    documents = []
    ids = []
    metadatas = []
    
    # --- STEP A: READ PDFs ---
    for filename in files:
        file_path = os.path.join(materials_dir, filename)
        print(f"    Processing: {filename}...", end="")
        
        try:
            reader = PdfReader(file_path)
            pages_added = 0
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Only save pages that actually have text
                if text and len(text) > 50:
                    documents.append(text)
                    ids.append(f"{filename}_p{page_num}")
                    metadatas.append({"source": filename, "page": page_num})
                    pages_added += 1
            
            print(f" Done! ({pages_added} pages)")
            
        except Exception as e:
            print(f" Failed to read PDF: {e}")

    # --- STEP B: UPLOAD TO DATABASE ---
    if documents:
        print(f"\n--- Embeddings in progress ({len(documents)} pages)... ---")
        
        # Batch process to be safe
        batch_size = 20
        total_batches = (len(documents) // batch_size) + 1
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            
            collection.upsert(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta
            )
            print(f"   Batch {i//batch_size + 1}/{total_batches} Saved.")
            time.sleep(0.5) # Polite pause for API

        print("\n" + "="*50)
        print(f"✅ SUCCESS! RAG Database built in 'my_knowledge_base'")
        print("="*50)
        
    else:
        print("⚠️ No text found. Please check your PDFs.")

if __name__ == "__main__":
    build_knowledge_base()