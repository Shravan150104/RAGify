import subprocess
import json
from pathlib import Path

# === Load video_id and config ===
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

VIDEO_ID = config["video_id"]
SKIP_OCR = config.get("skip_ocr", False)

print(f"\n🚀 Starting RAGify pipeline for: {VIDEO_ID}\n")

# Step 1: Check if transcript exists
transcript_path = Path(f"data/transcripts/{VIDEO_ID}.json")
if not transcript_path.exists():
    print(f"❌ Transcript not found: {transcript_path}")
    print("📝 Please run WhisperX manually and save output there.")
    exit(1)

# Step 2: OCR (if not skipped)
if not SKIP_OCR:
    print("\n🔎 Running OCR...")
    subprocess.run(["python", "app/ocr_extractor.py"], check=True)
else:
    print("\n⚠️ Skipping OCR as per config...")

# Step 3: Chunking
print("\n✂️ Chunking transcript + OCR...")
subprocess.run(["python", "app/chunker.py"], check=True)

# Step 4: Embedding
print("\n🔗 Embedding chunks into ChromaDB...")
subprocess.run(["python", "app/embedder.py"], check=True)

# Step 5: QA CLI (optional interactive)
print("\n💬 Launching QA CLI... (type 'exit' to quit)")
subprocess.run(["python", "app/qa.py"])
