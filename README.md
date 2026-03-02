# Smart Contract Summary & Q&A Assistant

Upload your PDF or DOCX contracts and ask questions about them. The app retrieves relevant sections from the document and uses an LLM to generate grounded answers — so you get real information, not hallucinations.

Built with LangChain, FAISS, Gradio, LangServe, and FastAPI.

---

## How to Set Up and Run

### Step 1: Clone and enter the project

```bash
cd smart-contract-assistant
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
# .venv\Scripts\activate    # on Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything: LangChain, FAISS, Gradio, FastAPI, file parsers, etc.

### Step 4: Set up your API keys

Copy the example env file:

```bash
cp .env.example .env
```

Then open `.env` and fill in your keys:

- `DEEPSEEK_API_KEY` — get one from [platform.deepseek.com](https://platform.deepseek.com/)
- `GOOGLE_API_KEY` — get one from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

The other settings (model names, chunk size, etc.) have sensible defaults, so you can leave them as-is.

### Step 5: Start the backend server

```bash
python server.py
```

This starts the FastAPI server on `http://localhost:9012`. Keep this terminal open.

### Step 6: Start the Gradio UI

Open a second terminal (make sure you activate the venv again) and run:

```bash
python app.py
```

Then open `http://localhost:7860` in your browser. That's it — you should see the app.

---

## How to Use It

1. Go to the **Upload Document** tab, pick a PDF or DOCX file, and click "Process Document"
2. Wait for the success message (it chunks and embeds the document behind the scenes)
3. Switch to the **Chat** tab and ask questions like "What are the payment terms?" or "Who are the parties?"
4. The **Summary** tab gives you a quick overview of the whole document
5. The **Evaluate** tab lets you test how accurate the Q&A pipeline is (more on this below)

You can upload multiple documents — they all get merged into the same vector store.

---

## File Structure

```
smart-contract-assistant/
├── config.py          # Loads API keys and settings from .env, provides get_llm() and get_embedder()
├── ingest.py          # Document ingestion: load PDF/DOCX → chunk → embed → save to FAISS
├── rag_chain.py       # The RAG pipeline: retrieval + LLM answering + summarization
├── app.py             # Gradio frontend (Upload, Chat, Summary, Evaluate tabs)
├── server.py          # FastAPI backend with REST endpoints + LangServe routes
├── evaluation.py      # LLM-as-a-Judge evaluation pipeline
├── requirements.txt   # Python dependencies
├── .env.example       # Template for API keys
├── .env               # Your actual API keys (not committed to git)
├── vectorstore/       # FAISS index files (auto-created after first upload)
└── uploads/           # Uploaded documents get saved here
```

**Quick overview of what each file does:**

- **`config.py`** — Central place for all settings. Reads API keys and model names from `.env`. Has two factory functions: `get_llm()` returns a DeepSeek chat model, `get_embedder()` returns a Google Gemini embedding model. Every other file imports from here.

- **`ingest.py`** — Handles the whole ingestion flow. `process_file()` is the main function: it loads the document (PyMuPDF for PDFs, docx2txt for DOCX files), splits it into chunks using `RecursiveCharacterTextSplitter`, embeds those chunks, and saves them into a FAISS vector store. If a store already exists, it merges the new chunks in.

- **`rag_chain.py`** — Where the actual Q&A logic lives. When you ask a question, it searches the vector store for the top 4 most relevant chunks, reorders them using `LongContextReorder` (so the best ones aren't buried in the middle), builds a prompt with the context, and streams the LLM's response token by token. It also has a relevance guard — if your first question is off-topic, it politely redirects you.

- **`app.py`** — The Gradio UI. It talks to the FastAPI backend over HTTP (doesn't call the RAG functions directly). Has four tabs: Upload, Chat, Summary, and Evaluate. The chat streams responses in real time so you see the answer being typed out.

- **`server.py`** — FastAPI server exposing everything as REST endpoints: `/upload`, `/qa_stream`, `/summarize`, `/evaluate`. Also sets up two LangServe routes (`/qa` and `/retriever`) for programmatic access. Runs on port 9012.

- **`evaluation.py`** — Automated testing of the RAG pipeline using the LLM-as-a-Judge method (explained below).

---

## How the Evaluation Works

The evaluation tab uses an **LLM-as-a-Judge** approach to test whether the RAG pipeline gives correct answers. Here's the process step by step:

### Step 1: Generate synthetic test questions

The system picks random pairs of document chunks from the vector store and asks the LLM to generate a question and a "ground truth" answer based on those chunks. This gives us test cases that are actually grounded in the uploaded documents.

### Step 2: Get RAG answers

Each generated question is then fed through the normal RAG pipeline (retrieve → build prompt → ask LLM), just like a real user would. The RAG answer is saved for comparison.

### Step 3: Judge the answers

A separate LLM call compares the RAG answer against the ground truth. The judge scores each one:

- **[1]** — The RAG answer is wrong or missing important information
- **[2]** — The RAG answer is essentially correct (same meaning as the ground truth)
- **[3]** — The RAG answer is correct and adds useful extra detail

### Step 4: Report

The final report shows the overall accuracy (% of answers scoring 2 or 3) plus per-question breakdowns with the question, ground truth, RAG answer, and the judge's reasoning.

You can control how many test questions to generate with the slider (1–10). More questions = more reliable results but takes longer.

**Why this approach?** Simple keyword matching doesn't work well because two answers can say the same thing in completely different words. Using an LLM as a judge handles paraphrasing and semantic equivalence much better.

---

## Tech Stack

- **LLM**: DeepSeek (via OpenAI-compatible API)
- **Embeddings**: Google Gemini (`gemini-embedding-001`)
- **Vector Store**: FAISS
- **Framework**: LangChain (LCEL chains)
- **Frontend**: Gradio
- **Backend**: FastAPI + LangServe

---

## API Endpoints (server.py)

If you want to use the backend directly (without the Gradio UI):

| Endpoint      | Method | What it does                              |
|---------------|--------|-------------------------------------------|
| `/upload`     | POST   | Upload and process a PDF/DOCX file        |
| `/qa_stream`  | POST   | Ask a question, get a streaming response  |
| `/summarize`  | POST   | Get a summary of all uploaded documents   |
| `/evaluate`   | POST   | Run the LLM-as-a-Judge evaluation         |
| `/qa`         | POST   | LangServe route for Q&A                   |
| `/retriever`  | POST   | LangServe route for raw chunk retrieval   |
