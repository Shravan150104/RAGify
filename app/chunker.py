import json
import os
from pathlib import Path

# === Load video_id from config.json ===
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH) as f:
    VIDEO_ID = json.load(f)["video_id"]

# === Paths based on VIDEO_ID ===
TRANSCRIPT_PATH = Path(__file__).parent.parent / f"data/transcripts/{VIDEO_ID}.json"
OCR_JSON_PATH   = Path(__file__).parent.parent / f"data/ocr_texts/{VIDEO_ID}.json"
OUTPUT_PATH     = Path(__file__).parent.parent / f"chunks/{VIDEO_ID}.json"

CHUNK_DURATION = 15
CHUNK_STRIDE = 5

# === Load transcript ===
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)["segments"]

# === Load OCR ===
with open(OCR_JSON_PATH, "r", encoding="utf-8") as f:
    ocr_data = json.load(f)

# === Build timestamp-to-OCR mapping ===
ocr_index = {}
frame_rate = 1 / 3  # frames every 3s

for frame_name, texts in ocr_data.items():
    frame_idx = int(Path(frame_name).stem.split("_")[1])
    timestamp = frame_idx * 3  # estimate time
    ocr_index[timestamp] = texts

# === OCR text lookup ===
def get_ocr_text(start, end):
    matched = []
    for ts, texts in ocr_index.items():
        if start <= ts <= end:
            matched.extend(texts)
    return list(set(matched))

# === Chunking ===
chunks = []
start = 0
end_time = transcript[-1]["end"]

while start < end_time:
    end = start + CHUNK_DURATION
    chunk_text = [
        seg["text"]
        for seg in transcript
        if seg["start"] >= start and seg["end"] <= end
    ]
    ocr_texts = get_ocr_text(start, end)
    ocr_str = " OCR: " + ", ".join([f"'{t}'" for t in ocr_texts]) if ocr_texts else ""

    if chunk_text:
        chunks.append({
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "text": " ".join(chunk_text) + ocr_str
        })

    start += CHUNK_STRIDE

# === Save ===
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Saved {len(chunks)} chunks to {OUTPUT_PATH}")
