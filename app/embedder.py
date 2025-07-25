import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === Load video_id from config.json ===
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH) as f:
    VIDEO_ID = json.load(f)["video_id"]

# === Paths ===
CHUNKS_PATH = Path(__file__).parent.parent / f"chunks/{VIDEO_ID}.json"
DB_DIR = Path(__file__).parent.parent / "db"
COLLECTION_NAME = "ragify_chunks"

# === Load chunks ===
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# === Load embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Setup ChromaDB ===
client = chromadb.PersistentClient(path=str(DB_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# === Clear previous entries (only for this video_id) ===
all_ids = collection.get()["ids"]
video_ids_to_delete = [cid for cid in all_ids if cid.startswith(VIDEO_ID)]
if video_ids_to_delete:
    collection.delete(ids=video_ids_to_delete)
    print(f"🧹 Cleared {len(video_ids_to_delete)} entries for {VIDEO_ID}")

# === Add chunks ===
for i, chunk in enumerate(chunks):
    chunk_text = chunk["text"]
    embedding = model.encode(chunk_text).tolist()
    metadata = {
        "start_time": chunk["start_time"],
        "end_time": chunk["end_time"],
        "video_id": VIDEO_ID
    }

    collection.add(
        documents=[chunk_text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[f"{VIDEO_ID}_chunk_{i}"]
    )

print(f"✅ Embedded {len(chunks)} chunks for video_id '{VIDEO_ID}' into ChromaDB")
