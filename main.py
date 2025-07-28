import os
import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer, util

# Folder paths
INPUT_DIR = "sample_input"
OUTPUT_DIR = "sample_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "semantic_results.json")

# Load PDF files from input folder
pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

if not pdf_files:
    print("❌ No PDF files found in 'sample_input'.")
    exit()

print(f"🔍 Found {len(pdf_files)} PDF(s) in '{INPUT_DIR}'.")

# Extract paragraphs with metadata
all_data = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(INPUT_DIR, pdf_file)
    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4].strip()
            if len(text.split()) >= 5:
                all_data.append({
                    "pdf": pdf_file,
                    "page": page_num + 1,
                    "text": text
                })

print(f"✅ Extracted {len(all_data)} paragraphs from {len(pdf_files)} PDFs.")

# Convert paragraphs to sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
paragraphs = [d["text"] for d in all_data]
para_embeddings = model.encode(paragraphs, convert_to_tensor=True)

# Get user query
query = input("Enter a topic to search (e.g., AI, insurance, pesticide): ").strip()
query_embedding = model.encode(query, convert_to_tensor=True)

# Perform semantic search
top_k = 10
hits = util.semantic_search(query_embedding, para_embeddings, top_k=top_k)[0]

# Format top results
results = []
for hit in hits:
    data = all_data[hit['corpus_id']]
    results.append({
        "pdf": data["pdf"],
        "page": data["page"],
        "text": data["text"],
        "score": round(hit["score"], 4)
    })

# Save output
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# Display results
for r in results:
    print(f"\n📘 {r['pdf']} — Page {r['page']} — Score: {r['score']}")
    print(f"→ {r['text'][:300]}...\n")

print(f"\n✅ Results saved to: {OUTPUT_FILE}")
