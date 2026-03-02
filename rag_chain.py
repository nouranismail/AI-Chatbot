# RAG chain: handles retrieval, relevance checking, Q&A, and summarization

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_transformers import LongContextReorder

from ingest import load_vectorstore
from config import get_llm



RAG_SYSTEM_PROMPT = """You are a helpful Smart Contract Assistant.
Answer the user's question based on the context provided below and the conversation history.
If the user refers to something discussed earlier, use the conversation history to understand what they mean.
If the answer is not in the context, say: 'I don't have enough information to answer that based on the uploaded documents.'
Always mention which source you used.

Context:
{context}"""

OFFTOPIC_SYSTEM_PROMPT = """You are a helpful Smart Contract Assistant.
The user's question does not seem related to the uploaded documents.
Politely explain that you can only answer questions about the uploaded documents,
and suggest they ask something related to the document content."""

SUMMARY_SYSTEM_PROMPT = """You are a document summarization assistant.
Provide a clear and concise summary of the document content below.
Highlight the key points, important clauses, and any notable terms.

Document Content:
{context}"""

RELEVANCE_CHECK_PROMPT = """You are a relevance checker.
Given the following context from uploaded documents, conversation history, and a user question,
decide if the question is related to the document content or is a follow-up to the conversation.

If the user is asking a follow-up question (like "tell me more", "explain that", "what else"), reply "relevant".

Reply with ONLY one word: "relevant" or "offtopic"

Context (first 500 chars):
{context}

Conversation history:
{chat_history}

User question:
{question}"""



def docs_to_text(docs):

    if not docs:
        return "No relevant documents found."

    text = ""
    for doc in docs:
        source = doc.metadata.get("source", "Document")
        text += f"[From: {source}]\n{doc.page_content}\n\n"
    return text


def retrieve_context(question, k=4):

    vector_store = load_vectorstore()
    if vector_store is None:
        return "No documents uploaded yet."

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    chunks = retriever.invoke(question)


    reordered = LongContextReorder().transform_documents(chunks)
    return docs_to_text(reordered)


def is_question_relevant(question, context, chat_history=None):
    # Quick relevance check — returns True if the question is about the docs
    llm = get_llm()

    history_str = "No previous conversation."
    if chat_history:
        lines = []
        for role, content in chat_history:
            label = "User" if role in ("user", "human") else "Assistant"
            lines.append(f"{label}: {content}")
        history_str = "\n".join(lines[-6:])

    prompt = RELEVANCE_CHECK_PROMPT.format(
        context=context[:500],
        chat_history=history_str,
        question=question,
    )

    response = llm.invoke(prompt).content.strip().lower()
    return "relevant" in response



def ask_question(question):

    context = retrieve_context(question)

    if context == "No documents uploaded yet.":
        return "Please upload a document first before asking questions."

    if not is_question_relevant(question, context):
        prompt = ChatPromptTemplate.from_messages([
            ("system", OFFTOPIC_SYSTEM_PROMPT),
            ("user", "{input}"),
        ])
        chain = prompt | get_llm() | StrOutputParser()
        return chain.invoke({"input": question})

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("user", "{input}"),
    ])
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"input": question, "context": context})


def ask_question_stream(question, chat_history=None):
    """Ask a question and stream the answer token by token.
    chat_history is a list of (role, content) tuples.
    """
    context = retrieve_context(question)

    if context == "No documents uploaded yet.":
        yield "Please upload a document first before asking questions."
        return

    # only check relevance on the first message
    if not chat_history and not is_question_relevant(question, context):
        prompt = ChatPromptTemplate.from_messages([
            ("system", OFFTOPIC_SYSTEM_PROMPT),
            ("user", "{input}"),
        ])
        chain = prompt | get_llm() | StrOutputParser()
        for token in chain.stream({"input": question}):
            yield token
        return

    messages = [SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))]

    if chat_history:
        for role, content in chat_history:
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=question))

    llm = get_llm()
    for token in llm.stream(messages):
        yield token.content


def summarize_document():

    vector_store = load_vectorstore()
    if vector_store is None:
        return "No document uploaded yet. Please upload a document first."

    all_chunks = list(vector_store.docstore._dict.values())
    context = docs_to_text(all_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("user", "Please summarize this document."),
    ])
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": context})
