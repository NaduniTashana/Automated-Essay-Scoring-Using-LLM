import os
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import time

# --- 1. CONFIGURATION (NO API KEY NEEDED) ---
# We use the 'all-MiniLM-L6-v2' model.
# It will download automatically the first time you run this (approx. 80MB).
print("⏳ Loading local embedding model (all-MiniLM-L6-v2)...")
local_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded! Running on your CPU (Free).")

# --- 2. DEFINE LOCAL EMBEDDING FUNCTION ---
# This class tells ChromaDB how to use your local model instead of Google's
class LocalEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # model.encode() turns text into a list of numbers (vectors)
        # convert_to_numpy=True ensures compatibility with Chroma
        embeddings = local_model.encode(input, convert_to_numpy=True).tolist()
        return embeddings

# --- 3. INITIALIZE DATABASE ---
chroma_client = chromadb.PersistentClient(path="my_knowledge_base_local") # New folder for local DB
embedding_function = LocalEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="lecture_notes",
    embedding_function=embedding_function
)

def build_knowledge_base():
    materials_dir = "materials"
    
    if not os.path.exists(materials_dir):
        print(f"❌ Error: Folder '{materials_dir}' not found.")
        print("   Please create a folder named 'materials' and put your PDFs inside.")
        return

    print(f"--- 📂 Scanning '{materials_dir}' for PDFs ---")
    
    documents = []
    ids = []
    metadatas = []
    
    files = [f for f in os.listdir(materials_dir) if f.lower().endswith(".pdf")]
    
    if not files:
        print("❌ No PDF files found!")
        return

    print(f"   Found {len(files)} files. Starting Ingestion...")

    for filename in files:
        file_path = os.path.join(materials_dir, filename)
        print(f"   📖 Reading: {filename}...", end="")
        
        try:
            reader = PdfReader(file_path)
            file_pages = 0
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Filter out empty pages
                if text and len(text) > 50: 
                    documents.append(text)
                    ids.append(f"{filename}_p{page_num}") 
                    metadatas.append({"source": filename, "page": page_num})
                    file_pages += 1
            
            print(f" Done! ({file_pages} pages)")
            
        except Exception as e:
            print(f" ❌ Failed: {e}")

    # --- 4. SAVE TO DATABASE ---
    if documents:
        print(f"\n--- 🧠 Embeddings in progress (Local CPU)... ---")
        
        # We process in batches to keep your RAM usage low
        batch_size = 20
        total_batches = len(documents) // batch_size + 1
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            
            collection.upsert(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_meta
            )
            print(f"   ✅ Processed batch {i//batch_size + 1}/{total_batches}")

        print("\n" + "="*50)
        print(f"✅ SUCCESS! Local RAG Database built in 'my_knowledge_base_local'")
        print("   - Cost: $0.00")
        print("   - Privacy: 100% Local")
        print("="*50)
        
        # Test Query
        results = collection.query(
            query_texts=["What is usability?"],
            n_results=1
        )
        print(f"🔎 Test Retrieval: Found {len(results['ids'][0])} document.")
        
    else:
        print("⚠️ No valid text found in PDFs.")

if __name__ == "__main__":
    build_knowledge_base()