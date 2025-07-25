# 🎬 RAGify – Civic Video Question Answering Agent

RAGify is an intelligent pipeline for civic meeting video understanding. It transcribes, extracts text from video frames, combines both sources, and enables question-answering using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🧠 Transcript-based Q&A using local LLMs (Ollama + Mistral)
- 🖼️ OCR from video frames (for on-screen data like dates or slides)
- 🧩 Chunked transcript + OCR fusion for better context
- 🔍 Vector-based retrieval with Sentence Transformers + ChromaDB
- 💬 Interactive Q&A CLI
- 📦 Auto-generated summaries (optional)
- 🔄 Scalable to multiple videos (just drop into `/data/videos`)

---

## 📁 Project Structure
```bash
ragify/
│
├── app/
│ ├── run_pipeline.py # Full pipeline (transcribe → ocr → chunk → embed → QA)
│ ├── chunker.py # Combines transcript + OCR → JSON chunks
│ ├── embedder.py # Embeds chunks into ChromaDB
│ ├── ocr_extractor.py # OCR from video frames using EasyOCR
│ ├── qa.py # CLI Q&A interface
│ ├── retriever.py # Chunk retriever logic
│ └── assets/ # Logo and static assets
│
├── data/
│ ├── videos/ # Raw videos (video1.mp4, video2.webm)
│ ├── frames/ # Extracted frames from videos
│ ├── ocr_texts/ # OCR results (video1.json, ...)
│ └── transcripts/ # Whisper transcripts (video1.json, ...)
│
├── chunks/ # Final chunked data (video1.json, ...)
├── db/ # ChromaDB vector store
├── embeddings/ # Optional saved embeddings
├── export/ # Exported session summaries & QAs
├── requirements.txt
└── README.md

```
## 🔧 Setup
```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ragify.git
cd ragify

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download model weights
# If using Ollama:
ollama pull mistral
```
📽️ Usage Guide
1. 📁 Add your video
```bash
Place your .mp4 / .webm video in data/videos/ folder and name it like:
video1.mp4
video2.webm
Update VIDEO_ID in run_pipeline.py to match the video filename (without extension).
```
2. ▶️ Run full pipeline
```bash
python app/run_pipeline.py
```
This will:
- Run Whisper transcription
- Extract frames and apply OCR
- Create transcript + OCR chunks
- Embed into ChromaDB
- Launch the Q&A CLI

3. ❓ Ask Questions
```bash
Inside the CLI:
Ask: What did the speaker say about housing policy?
Ask: summary
Ask: exit
```
🧠 Tech Stack:

- Whisper (ASR)
- EasyOCR (image-to-text)
- entenceTransformers (all-MiniLM-L6-v2)
- ChromaDB (vector DB)
- Ollama (local LLMs),
- Python CLI (no frontend yet)

📌 TODO (Upcoming)

1. React frontend like Groq Chat,
2. Admin dashboard for summary & QA logs,
3. Multi-video search,
4. OCR post-processing for better accuracy
