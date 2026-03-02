# Document ingestion: load -> chunk -> embed -> store in FAISS

import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import get_embedder, CHUNK_SIZE, CHUNK_OVERLAP, VECTORSTORE_DIR


def load_document(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or DOCX.")

    docs = loader.load()
    print(f"Loaded {len(docs)} page(s) from {os.path.basename(file_path)}")
    return docs


def chunk_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", ";", ",", " "],
    )

    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_vectorstore(chunks):

    embedder = get_embedder()
    vectorstore = FAISS.from_documents(chunks, embedder)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to {VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore():

    index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_file):
        print("No saved vector store found.")
        return None

    embedder = get_embedder()
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedder,
        allow_dangerous_deserialization=True,
    )
    print("Loaded vector store from disk.")
    return vectorstore


def process_file(file_path):
    """Runs the full ingestion pipeline on a file and returns a status message."""
    try:
        docs = load_document(file_path)
        chunks = chunk_documents(docs)

        if len(chunks) == 0:
            return "Error: No text found in the document."


        existing_store = load_vectorstore()

        if existing_store is not None:
            embedder = get_embedder()
            new_store = FAISS.from_documents(chunks, embedder)
            existing_store.merge_from(new_store)
            existing_store.save_local(VECTORSTORE_DIR)
            total = len(existing_store.docstore._dict)
            msg = f"Added {len(chunks)} chunks to existing store (total: {total} chunks)"
        else:
            create_vectorstore(chunks)
            msg = f"Created new vector store with {len(chunks)} chunks"

        print(msg)
        return f"Success! {msg}"

    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        print(error_msg)
        return error_msg
