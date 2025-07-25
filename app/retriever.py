import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# === Load video_id from config.json ===
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    VIDEO_ID = json.load(f)["video_id"]

# === Chroma setup ===
DB_DIR = Path(__file__).parent.parent / "db"
COLLECTION_NAME = "ragify_chunks"

client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Ask your question
question = input("❓ Question: ").strip()
query_emb = model.encode(question).tolist()

# === Query only chunks for this video_id
results = collection.query(
    query_embeddings=[query_emb],
    n_results=3,
    where={"video_id": VIDEO_ID},
    include=["documents", "metadatas"]
)

# === Show top results
print("\n📥 Top Relevant Chunks:\n")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"[{meta['start_time']} – {meta['end_time']}]")
    print(doc)
    print("-" * 50)
