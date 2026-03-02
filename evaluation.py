# Evaluates the RAG pipeline using an LLM-as-a-Judge approach

import json
import random

from rag_chain import ask_question
from ingest import load_vectorstore
from config import get_llm



QA_GENERATION_PROMPT = """Use the two documents below to write one question and one answer.
Format your response exactly like this (no extra text):
Question: <your question here>

Answer: <your answer here>

Document 1:
{doc1}

Document 2:
{doc2}"""

JUDGE_PROMPT = """You are checking if two answers say the same thing.
The first answer is the ground truth. The second is from a RAG system.

Rules:
- Different wording is fine.
- Extra detail is fine.
- Only give [1] if the RAG answer is factually wrong or missing a key fact.

Scores:
[1] RAG answer is wrong or missing a key fact.
[2] RAG answer means the same as the ground truth.
[3] RAG answer is correct and adds useful extra detail.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG Answer: {rag_answer}

Your score and reason:"""



def run_evaluation_stream(num_questions=3):
    """Yields progress lines as it runs. Last line has the JSON report."""
    vector_store = load_vectorstore()
    if vector_store is None:
        yield "No vector store found. Please upload a document first."
        return

    all_chunks = list(vector_store.docstore._dict.values())
    if len(all_chunks) < 2:
        yield "Need at least 2 document chunks to run evaluation."
        return

    llm = get_llm()
    yield f"Starting evaluation with {num_questions} test questions...\n"

    # Step 1: Generate synthetic Q&A pairs
    questions = []
    ground_truths = []

    yield "--- Step 1/3: Generating synthetic Q&A pairs ---\n"

    for i in range(num_questions):
        yield f"Generating Q&A pair {i + 1}/{num_questions}..."

        chunk1, chunk2 = random.sample(all_chunks, 2)
        prompt = QA_GENERATION_PROMPT.format(
            doc1=chunk1.page_content,
            doc2=chunk2.page_content,
        )

        response = llm.invoke(prompt).content
        parts = response.split("\n\n", 1)
        question_part = parts[0].strip()
        answer_part = parts[1].strip() if len(parts) > 1 else ""

        questions.append(question_part)
        ground_truths.append(answer_part)
        yield f"Done: {question_part[:80]}...\n"

    # Step 2: Get RAG answers
    rag_answers = []

    yield "\n--- Step 2/3: Getting RAG answers ---\n"

    for i, question in enumerate(questions):
        yield f"Asking RAG question {i + 1}/{num_questions}..."

        rag_answer = ask_question(question)
        rag_answers.append(rag_answer)
        yield f"Done: {rag_answer[:80]}...\n"

    # Step 3: Judge each answer
    scores = []
    judgments = []

    yield "\n--- Step 3/3: Judging answers ---\n"

    for i in range(num_questions):
        yield f"Judging answer {i + 1}/{num_questions}..."

        prompt = JUDGE_PROMPT.format(
            question=questions[i],
            ground_truth=ground_truths[i],
            rag_answer=rag_answers[i],
        )

        judgment = llm.invoke(prompt).content
        scores.append(judgment)
        judgments.append(judgment)

        if "[3]" in judgment:
            label = "[3] Correct + extra detail"
        elif "[2]" in judgment:
            label = "[2] Correct"
        else:
            label = "[1] Wrong or missing info"

        yield f"Result: {label}\n"

    # calculate metrics
    passed = sum(1 for s in scores if "[2]" in s or "[3]" in s)
    excellent = sum(1 for s in scores if "[3]" in s)
    failed = num_questions - passed
    accuracy = passed / len(scores) if scores else 0

    yield "\n✅ Evaluation complete! Generating report...\n"


    details = []
    for i in range(num_questions):
        if "[3]" in scores[i]:
            score_val = 3
        elif "[2]" in scores[i]:
            score_val = 2
        else:
            score_val = 1

        details.append({
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "rag_answer": rag_answers[i][:500],
            "judgment": judgments[i],
            "score": score_val,
        })

    report = {
        "num_questions": num_questions,
        "passed": passed,
        "excellent": excellent,
        "failed": failed,
        "accuracy": round(accuracy * 100, 1),
        "details": details,
    }

    yield ":::REPORT:::" + json.dumps(report)



def run_evaluation(num_questions=3):

    report = None
    for line in run_evaluation_stream(num_questions):
        print(line)
        if line.startswith(":::REPORT:::"):
            report = json.loads(line.replace(":::REPORT:::", ""))
    return report


if __name__ == "__main__":
    run_evaluation(num_questions=3)
