import os
import json
import re
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# === Load config ===
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)



VIDEO_ID = config["video_id"]
CHUNKS_PATH = f"chunks/{VIDEO_ID}.json"
DB_PATH = "db"
N_RESULTS = 3

# === Setup ===
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="ragify_chunks")


# === Utilities ===
def timestamp_to_seconds(ts):
    try:
        return int(float(ts))
    except:
        return 0


def linkify_timestamps(answer_text):
    matches = re.findall(r"\[(\d+\.?\d*)\s*–\s*(\d+\.?\d*)\]", answer_text)
    for start, end in matches:
        seconds = timestamp_to_seconds(start)
        link = f"https://youtube.com/watch?v={VIDEO_ID}&t={seconds}s"
        markdown_link = f"[{start} – {end}](🔗 {link})"
        answer_text = answer_text.replace(f"[{start} – {end}]", markdown_link)
    return answer_text


def build_prompt(question, chunks):
    prompt = (
        "You are a helpful assistant that answers civic video questions using transcript + OCR.\n"
        "Always cite time ranges in the format [start – end].\n\n"
        "Relevant transcript chunks:\n"
    )
    for chunk in chunks:
        prompt += f"[{chunk['start_time']} – {chunk['end_time']}] {chunk['text']}\n"
    prompt += f"\nQuestion: {question}\nAnswer:"
    return prompt


def generate_summary():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Strip OCR text
    cleaned_text = []
    for chunk in chunks:
        text = chunk["text"]
        if "OCR:" in text:
            text = text.split("OCR:")[0].strip()
        cleaned_text.append(text)

    full_text = " ".join(cleaned_text)

    prompt = (
        "You are a helpful assistant. Summarize this civic meeting transcript in 5 bullet points:\n\n"
        + full_text
        + "\n\nSummary:"
    )
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]



# === Chat Loop ===
print("\n🧠 Civic Video Q&A (RAGify)")
print("Ask a question or type 'summary' to get a bullet-point overview.\n")

while True:
    user_input = input("❓ Ask: ").strip()
    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        break

    if user_input.lower() == "summary":
        print("\n📋 Summary:\n")
        summary = generate_summary()
        print(summary)
        continue

    # === Embed query
    query_emb = model.encode(user_input).tolist()
    result = collection.query(
        query_embeddings=[query_emb],
        n_results=N_RESULTS,
        include=["documents", "metadatas"]
    )
    retrieved_chunks = [
        {"start_time": m["start_time"], "end_time": m["end_time"], "text": d}
        for d, m in zip(result["documents"][0], result["metadatas"][0])
    ]

    # === LLM Answer
    prompt = build_prompt(user_input, retrieved_chunks)
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"]
    answer = linkify_timestamps(answer)

    # === Output
    print("\n🧠 Answer:\n")
    print(answer)

    print("\n📥 Top Relevant Chunks:\n")
    for chunk in retrieved_chunks:
        print(f"[{chunk['start_time']} – {chunk['end_time']}]")
        print(chunk["text"])
        print("-" * 50)
