# FastAPI backend with LangServe routes

import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from config import UPLOADS_DIR
from ingest import process_file
from rag_chain import ask_question, ask_question_stream, retrieve_context, summarize_document
from evaluation import run_evaluation_stream



app = FastAPI(
    title="Smart Contract Assistant API",
    version="1.0",
    description="REST API for the Smart Contract Assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[list] = None



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    save_path = os.path.join(UPLOADS_DIR, file.filename)
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    result = process_file(save_path)
    return JSONResponse(content={"message": result})


@app.post("/summarize")
def summarize():

    result = summarize_document()
    return JSONResponse(content={"summary": result})


@app.post("/qa_stream")
def qa_stream(req: QuestionRequest):


    def token_generator():
        for token in ask_question_stream(req.question, req.chat_history):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post("/evaluate")
def evaluate(num_questions: int = 3):


    def progress_generator():
        for line in run_evaluation_stream(num_questions=num_questions):
            yield line + "\n"

    return StreamingResponse(progress_generator(), media_type="text/plain")


# LangServe

qa_runnable = RunnableLambda(ask_question)
add_routes(app, qa_runnable, path="/qa")

retriever_runnable = RunnableLambda(retrieve_context)
add_routes(app, retriever_runnable, path="/retriever")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9012)
