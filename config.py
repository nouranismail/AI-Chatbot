# Loads env vars and sets up the LLM + embedding model

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


def get_llm():
    """Returns a Google Gemini or DeepSeek LLM client based on LLM_MODEL."""
    if LLM_MODEL.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )


def get_embedder():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is empty. Check your .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
