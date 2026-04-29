# 🧠 OmniRAG — Multi-Modal Retrieval-Augmented Generation Research Assistant

> Query anything. Web pages, documents, videos, images — one intelligent interface powered by LLMs.

---

## 🚀 What is OmniRAG?

**OmniRAG** is a multi-modal AI research assistant that lets you ingest content from virtually any source — web URLs, PDFs, DOCX files, YouTube videos, and images — and ask natural language questions about them using Retrieval-Augmented Generation (RAG).

Built with LangChain, Groq's blazing-fast LLM inference, and a local vector store, OmniRAG extracts meaning from diverse content types and returns grounded, context-aware answers — no hallucinations, no guesswork.

---

## ✨ Features

| Mode | Input | How it works |
|------|-------|-------------|
| 🌐 **Web Links** | Up to 3 URLs | Scrapes and indexes page content |
| 📄 **PDF / DOCX** | Upload files | Extracts and chunks document text |
| 🎬 **YouTube Video** | Up to 3 video URLs | Fetches auto-generated or manual captions |
| 🖼️ **Image** | Upload images | Uses Groq Llama 4 Scout vision to extract all content |

- **Zero-cost inference** — Powered entirely by Groq's free tier (no OpenAI, no Anthropic billing)
- **Local embeddings** — `Alibaba-NLP/gte-base-en-v1.5` runs on your machine, no embedding API needed
- **Smart context budgeting** — Chunks are ranked by semantic distance and selected within token limits
- **Clean Streamlit UI** — Tabbed interface, file uploaders, image previews, source attribution

---

## 🏗️ Architecture

```
User Input (URL / File / Video / Image)
        │
        ▼
┌─────────────────────────────────┐
│         Source Loaders          │
│  requests + UnstructuredHTML    │
│  PyPDFLoader / Docx2txtLoader   │
│  YouTubeTranscriptApi           │
│  Groq Vision (Llama 4 Scout)    │
└────────────┬────────────────────┘
             │  raw text
             ▼
┌─────────────────────────────────┐
│     RecursiveCharacterSplitter  │
│     chunk_size=800, overlap=80  │
└────────────┬────────────────────┘
             │  chunks
             ▼
┌─────────────────────────────────┐
│   HuggingFace Embeddings        │
│   gte-base-en-v1.5 (local)      │
└────────────┬────────────────────┘
             │  vectors
             ▼
┌─────────────────────────────────┐
│       ChromaDB (local)          │
│   Persistent vector store       │
└────────────┬────────────────────┘
             │  similarity search (top-k)
             ▼
┌─────────────────────────────────┐
│   Token Budget Selector         │
│   Ranks chunks by distance,     │
│   fits within context window    │
└────────────┬────────────────────┘
             │  context
             ▼
┌─────────────────────────────────┐
│   Groq LLM                      │
│   llama-3.3-70b-versatile       │
└────────────┬────────────────────┘
             │
             ▼
         Answer + Sources
```

---

## 📁 Project Structure

```
omnirag/
├── main.py              # Streamlit UI — tabbed interface for all 4 modes
├── link_rag.py          # Web URL ingestion and QA
├── pdf_rag.py           # PDF and DOCX ingestion and QA
├── video_rag.py         # YouTube transcript ingestion and QA
├── image_rag.py         # Image content extraction (Groq Vision) and QA
├── .env                 # API keys (not committed)
├── requirements.txt     # Python dependencies
└── resource/
    ├── vectorstore/         # Chroma DB for web links
    ├── vectorstore_pdf/     # Chroma DB for documents
    ├── vectorstore_video/   # Chroma DB for videos
    └── vectorstore_image/   # Chroma DB for images
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/omnirag.git
cd omnirag
```

### 2. Install dependencies

```bash
pip install streamlit python-dotenv \
  langchain langchain-community langchain-text-splitters \
  langchain-chroma langchain-groq langchain-huggingface \
  chromadb requests unstructured \
  pypdf docx2txt \
  youtube-transcript-api \
  groq sentence-transformers
```

### 3. Set up your `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 4. Run the app

```bash
streamlit run main.py
```

---

## 🔑 API Keys & Cost

| Service | Used for | Free? |
|---------|----------|-------|
| [Groq](https://console.groq.com) | LLM inference + Vision | ✅ Free tier |
| HuggingFace Embeddings | Vector embeddings | ✅ Runs locally |
| YouTube Transcript API | Caption extraction | ✅ No key needed |

**OmniRAG costs $0 to run.** Everything is on Groq's free tier or runs locally.

---

## 🧩 Models Used

| Task | Model |
|------|-------|
| Text generation / QA | `llama-3.3-70b-versatile` via Groq |
| Image understanding | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq |
| Embeddings | `Alibaba-NLP/gte-base-en-v1.5` via HuggingFace (local) |

---

## 💡 How to Use

### 🌐 Web Links
1. Open the **Web Links** tab
2. Paste up to 3 URLs in the sidebar inputs
3. Click **Process URLs**
4. Type your question and get an answer with sources

### 📄 PDF / DOCX
1. Open the **PDF / DOCX** tab
2. Upload one or more `.pdf` or `.docx` files
3. Click **Process Files**
4. Ask anything about the document content

### 🎬 YouTube Video
1. Open the **YouTube Video** tab
2. Paste up to 3 YouTube video URLs (must have captions enabled)
3. Click **Process Videos**
4. Ask questions about what was said in the videos

### 🖼️ Image
1. Open the **Image** tab
2. Upload one or more images (JPG, PNG, WEBP, GIF)
3. Click **Process Images** — Groq Vision extracts all text, data, and visual content
4. Ask questions about what's in the images

---

## 🛠️ Key Technical Decisions

**Why RAG over direct LLM prompting?**
RAG grounds answers in your actual source content, eliminates hallucinations for factual queries, and works within LLM context window limits by selecting only the most relevant chunks.

**Why Groq?**
Groq's LPU hardware delivers 300–1000 tokens/second — 5–10x faster than GPU-based providers — entirely free for prototyping and personal use.

**Why local embeddings?**
No API calls, no rate limits, no cost, and `gte-base-en-v1.5` performs on par with OpenAI's `text-embedding-3-small` on retrieval benchmarks.

**Why separate vector stores per mode?**
Each source type is indexed independently so you can process and query different content types without them interfering with each other.

---

## 📊 RAG Configuration

```python
CHUNK_SIZE        = 800    # characters per chunk
CHUNK_OVERLAP     = 80     # overlap between chunks
RETRIEVAL_K       = 10     # top-k chunks retrieved
LLM_CONTEXT_LIMIT = 4096   # total token budget
MAX_ANSWER_TOKENS = 500    # reserved for LLM answer
CONTEXT_BUDGET    = 3396   # tokens available for chunks
```

---

## 🙌 Acknowledgements

- [LangChain](https://langchain.com) — document loading, splitting, and vector store abstractions
- [Groq](https://groq.com) — ultra-fast LLM and vision inference
- [ChromaDB](https://trychroma.com) — local vector store
- [HuggingFace](https://huggingface.co) — open-source embedding model
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — YouTube caption extraction

---

## 📄 License

MIT License — free to use, modify, and distribute.
