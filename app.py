# Gradio frontend — talks to the FastAPI backend over HTTP

import json
import requests
import gradio as gr

SERVER_URL = "http://localhost:9012"



def handle_upload(file_path):

    if file_path is None:
        return "Please select a file to upload."

    with open(file_path, "rb") as f:
        response = requests.post(f"{SERVER_URL}/upload", files={"file": f})

    if response.status_code == 200:
        return response.json().get("message", "File processed successfully.")
    else:
        return f"Error: {response.status_code} - {response.text}"



def chat_gen(message, history):
    """Streams a response from the API, passing along conversation history."""
    if not message.strip():
        yield "Please type a question."
        return


    chat_history = []
    for turn in history:
        role = turn.get("role", "user")
        content_parts = turn.get("content", [])
        if isinstance(content_parts, list):
            text = " ".join(
                part.get("text", "") for part in content_parts if isinstance(part, dict)
            )
        else:
            text = str(content_parts)

        if role == "assistant":
            role = "ai"
        chat_history.append((role, text))




    buffer = ""
    response = requests.post(
        f"{SERVER_URL}/qa_stream",
        json={"question": message, "chat_history": chat_history},
        stream=True,
    )

    if response.status_code != 200:
        yield f"Error: {response.status_code} - {response.text}"
        return

    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            buffer += chunk
            yield buffer



def handle_summarize():

    response = requests.post(f"{SERVER_URL}/summarize")

    if response.status_code == 200:
        return response.json().get("summary", "No summary returned.")
    else:
        return f"Error: {response.status_code} - {response.text}"



def format_report(report):

    n = report["num_questions"]
    passed = report["passed"]
    failed = report["failed"]
    acc = report["accuracy"]
    details = report["details"]

    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Metrics")
    lines.append(f"- **Test Questions:** {n}")
    lines.append(f"- **Passed:** {passed}")
    lines.append(f"- **Failed:** {failed}")
    lines.append(f"- **Accuracy:** {acc}%")
    lines.append("")

    lines.append("## Per-Question Results")
    for i, d in enumerate(details):
        score = d["score"]
        if score == 3:
            badge = "Correct + Extra Detail"
        elif score == 2:
            badge = "Correct"
        else:
            badge = "Wrong / Missing Info"

        lines.append("")
        lines.append(f"### Question {i + 1}: {badge}")
        lines.append(f"**Q:** {d['question']}")
        lines.append(f"**Ground Truth:** {d['ground_truth'][:200]}...")
        lines.append(f"**RAG Answer:** {d['rag_answer'][:200]}...")
        lines.append(f"**Judgment:** {d['judgment']}")

    lines.append("")
    lines.append("## Known Limitations")
    lines.append("- Questions are generated synthetically from random chunk pairs, "
                 "which may not reflect real user queries.")
    lines.append("- The judge LLM may have biases and is not a perfect evaluator.")
    lines.append("- Small sample sizes (< 10 questions) produce high-variance accuracy scores.")
    lines.append("- The evaluation does not test retrieval quality separately from answer quality.")
    lines.append("- Results may vary between runs due to random chunk selection and LLM non-determinism.")

    return "\n".join(lines)


def handle_evaluate(num_questions):
    """Streams eval progress from the API, then renders the final report."""
    progress = ""

    try:
        response = requests.post(
            f"{SERVER_URL}/evaluate",
            params={"num_questions": int(num_questions)},
            stream=True,
            timeout=300,
        )

        if response.status_code != 200:
            yield f"Error: {response.status_code} - {response.text}", ""
            return

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue

            if line.startswith(":::REPORT:::"):
                report_json = line.replace(":::REPORT:::", "")
                report_data = json.loads(report_json)
                progress += "\nDone!\n"
                yield progress, format_report(report_data)
            else:
                progress += line + "\n"
                yield progress, "_Evaluation in progress..._"

    except requests.exceptions.Timeout:
        yield progress + "\nEvaluation timed out.", "Evaluation timed out. Try fewer questions."
    except Exception as e:
        yield progress + f"\nError: {str(e)}", f"Error: {str(e)}"



def create_app():


    with gr.Blocks(title="Smart Contract Assistant") as demo:

        gr.Markdown("# Nouran's RAG Chatbot")
        gr.Markdown("Upload your contracts (PDF/DOCX) and ask questions about them!")


        with gr.Tab("Upload Document"):
            gr.Markdown("### Upload a PDF or DOCX file to get started")

            file_input = gr.File(
                label="Select a file",
                file_types=[".pdf", ".docx"],
                type="filepath",
            )
            upload_btn = gr.Button("Process Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False, lines=3)

            upload_btn.click(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_status],
            )


        with gr.Tab("Chat"):
            gr.Markdown("### Ask questions about your uploaded documents")
            gr.ChatInterface(fn=chat_gen)


        with gr.Tab("Summary"):
            gr.Markdown("### Get a summary of the uploaded document")

            summarize_btn = gr.Button("Generate Summary", variant="primary")
            summary_output = gr.Textbox(
                label="Document Summary", interactive=False, lines=15
            )

            summarize_btn.click(
                fn=handle_summarize,
                inputs=[],
                outputs=[summary_output],
            )


        with gr.Tab("Evaluate"):
            gr.Markdown("### Evaluate the RAG Pipeline")
            gr.Markdown(
                "Runs the **LLM-as-a-Judge** evaluation: generates synthetic Q&A pairs, "
                "asks the RAG chain, and judges correctness. Watch the progress in real time!"
            )

            with gr.Row():
                num_questions_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Number of test questions",
                )
                evaluate_btn = gr.Button("Run Evaluation", variant="primary")

            eval_progress = gr.Textbox(
                label="Live Progress", interactive=False, lines=15
            )
            eval_report = gr.Markdown(
                value="_Click 'Run Evaluation' to start._",
                label="Evaluation Report",
            )

            evaluate_btn.click(
                fn=handle_evaluate,
                inputs=[num_questions_slider],
                outputs=[eval_progress, eval_report],
            )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
