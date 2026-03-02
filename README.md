# Smart Contract Summary & Q&A Assistant

Upload your PDF or DOCX contracts and ask questions about them.

The system retrieves relevant sections from the document and uses an LLM to generate grounded answers — so you get real information, not hallucinations.

Built with: LangChain • FAISS • Gradio • LangServe • FastAPI


--------------------------------------------------
How to Use It
--------------------------------------------------

1) Upload Document
- Go to the Upload Document tab
- Select a PDF or DOCX file
- Click "Process Document"

The system will:
- Chunk the document
- Generate embeddings
- Store them in FAISS

--------------------------------------------------

2) Chat With Your Document
- Switch to the Chat tab
- Ask questions like:

What are the payment terms?
Who are the parties?
What happens in case of termination?

The system:
- Retrieves relevant chunks
- Reorders context
- Feeds it to the LLM
- Streams the answer in real time

--------------------------------------------------

3) Summary Tab
Provides a quick overview of the entire document
Using an LLM-based summarization pipeline.

--------------------------------------------------

4) Evaluate Tab
Tests the accuracy of your RAG pipeline using an
LLM-as-a-Judge evaluation system.

You can:
- Generate synthetic test questions
- Automatically compare RAG answers with ground truth
- View accuracy metrics and detailed reasoning

--------------------------------------------------

Multiple Documents
--------------------------------------------------

You can upload multiple files.

All documents:
- Are chunked
- Embedded
- Stored in the same FAISS vector store
- Automatically merged

--------------------------------------------------

How to Set Up and Run
Step 1: Clone and enter the project
cd smart-contract-assistant
Step 2: Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
# .venv\Scripts\activate    # on Windows

Step 3: Install dependencies
pip install -r requirements.txt
This installs everything: LangChain, FAISS, Gradio, FastAPI, file parsers, etc.

Step 4: Set up your API keys
Copy the example env file:

cp .env.example .env
Then open .env and fill in your keys:

DEEPSEEK_API_KEY — get one from platform.deepseek.com
GOOGLE_API_KEY — get one from aistudio.google.com/apikey
The other settings (model names, chunk size, etc.) have sensible defaults, so you can leave them as-is.

Step 5: Start the backend server
python server.py
This starts the FastAPI server on http://localhost:9012. Keep this terminal open.

Step 6: Start the Gradio UI
Open a second terminal (make sure you activate the venv again) and run:

python app.py
Then open http://localhost:7860 in your browser. That's it — you should see the app.

File Structure
--------------------------------------------------

smart-contract-assistant/

│
├── config.py        -> Loads API keys & settings from .env
│                      Provides get_llm() and get_embedder()
│
├── ingest.py         -> Document ingestion pipeline
│                      PDF/DOCX → Chunk → Embed → Save to FAISS
│
├── rag_chain.py      -> RAG pipeline:
│                      Retrieval + LLM answering + Summarization
│
├── app.py            -> Gradio UI
│                      Tabs: Upload | Chat | Summary | Evaluate
│
├── server.py         -> FastAPI backend
│                      REST APIs + LangServe routes
│
├── evaluation.py      -> LLM-as-a-Judge evaluation pipeline
│
├── requirements.txt   -> Project dependencies
├── .env.example       -> API keys template
├── .env               -> Your API keys (not committed)
│
├── vectorstore/       -> FAISS index (auto-created)
└── uploads/            -> Uploaded documents

--------------------------------------------------

Core Components Explanation
--------------------------------------------------

config.py
- Central configuration file
- Reads API keys from .env
- Defines:
  get_llm() -> Returns DeepSeek model
  get_embedder() -> Returns Gemini embedding model

All other files import from here.

--------------------------------------------------

ingest.py
Handles the ingestion pipeline:

1) Load PDF or DOCX
2) Split into chunks (RecursiveCharacterTextSplitter)
3) Generate embeddings
4) Store them inside FAISS

If a vector store already exists → New documents are merged.

--------------------------------------------------

rag_chain.py
Contains the main Q&A logic:

- Searches top 4 relevant chunks
- Uses LongContextReorder to improve context quality
- Builds a structured prompt
- Streams LLM response token by token
- Includes a relevance guard to redirect off-topic questions

--------------------------------------------------

app.py (Frontend)
Built with Gradio

Tabs:
- Upload
- Chat
- Summary
- Evaluate

Chat supports real-time streaming responses.

--------------------------------------------------

server.py
FastAPI backend exposing:

/upload
/qa_stream
/summarize
/evaluate

Also exposes:
/qa
/retriever

as LangServe programmatic routes.

Runs on port 9012.

--------------------------------------------------

evaluation.py
Implements LLM-as-a-Judge evaluation.

Evaluation Process:

Step 1 – Generate Synthetic Questions
- Random document chunks are selected
- LLM generates:
  - Question
  - Ground truth answer

Step 2 – Run RAG Pipeline
Each question is answered using:
Retrieve → Prompt → LLM → Answer

Step 3 – Judge Answers
Another LLM compares:

Score 1 → Wrong or missing information
Score 2 → Correct
Score 3 → Correct + Extra useful detail

Step 4 – Generate Report
Shows:
- Overall accuracy
- Per-question breakdown
- Ground truth
- RAG answer
- Judge reasoning

You control how many test questions to generate (1–10).

--------------------------------------------------

Why LLM-as-a-Judge?
--------------------------------------------------

Traditional keyword matching fails because
answers can be phrased differently.

LLM-as-a-Judge evaluates semantic meaning,
making evaluation much more reliable.

--------------------------------------------------

Tech Stack
--------------------------------------------------

LLM: DeepSeek (OpenAI-compatible API)
Embeddings: Google Gemini (gemini-embedding-001)
Vector Store: FAISS
Framework: LangChain (LCEL)
Frontend: Gradio
Backend: FastAPI + LangServe
