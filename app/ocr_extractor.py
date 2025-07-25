import os
import cv2
import easyocr
import json
from pathlib import Path

# === Load VIDEO_ID from config ===
CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    VIDEO_ID = json.load(f)["video_id"]

# === Paths ===
VIDEO_PATH   = Path(__file__).parent.parent / f"data/videos/{VIDEO_ID}.mp4"
FRAME_DIR    = Path(__file__).parent.parent / f"data/frames/{VIDEO_ID}"
OCR_OUT_PATH = Path(__file__).parent.parent / f"data/ocr_texts/{VIDEO_ID}.json"
FRAME_RATE   = 1  # frames per second (1 FPS = every 1s)

# === Create output folders ===
FRAME_DIR.mkdir(parents=True, exist_ok=True)
OCR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Extract frames ===
cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened() or fps == 0:
    print(f"❌ Failed to open video or invalid FPS: {VIDEO_PATH}")
    exit(1)

interval = max(1, int(fps / FRAME_RATE))  # avoid division by 0

frame_count = 0
saved = 0

print(f"🎞️ Video FPS: {fps}, Extracting frames every {interval} frames (~1s intervals)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % interval == 0:
        frame_path = FRAME_DIR / f"frame_{saved:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved += 1
    frame_count += 1

cap.release()
print(f"✅ Extracted {saved} frames to {FRAME_DIR}")

# === Run OCR ===
reader = easyocr.Reader(['en'])
ocr_results = {}

for frame_file in sorted(FRAME_DIR.glob("*.jpg")):
    print(f"🔍 OCR on {frame_file.name}...")
    results = reader.readtext(str(frame_file))
    texts = [text for (_, text, conf) in results if conf > 0.4]
    ocr_results[frame_file.name] = texts

# === Save results ===
with open(OCR_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(ocr_results, f, indent=2)

print(f"📄 OCR results saved to: {OCR_OUT_PATH}")
