from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import shutil
import os

from langchain_groq import ChatGroq

from config import *

from ingestion.pdf_loader import load_pdfs
from ingestion.splitter import split_documents
from ingestion.vectorstore import get_vectorstore

from summarization.transformers_sum import summarize_chunks
from summarization.llm_sum import summarize_with_llm

from rag.retriever import retrieve_chunks
from rag.qa import answer_question

from utils.pdf_export import save_summary_pdf


# -------------------------
# Upload PDF
# -------------------------

def upload_pdf(files):

    # remove previous PDFs
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    os.makedirs(DATA_DIR)

    for file in files:
        shutil.copy(
            file.name,
            os.path.join(
                DATA_DIR,
                os.path.basename(file.name)
            )
        )

    return "PDF uploaded successfully."

# -------------------------
# Build Vector Database
# -------------------------

def build_index():

    docs = load_pdfs(DATA_DIR)

    chunks = split_documents(
        docs,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    get_vectorstore(
        chunks,
        DB_DIR,
        EMBEDDING_MODEL
    )

    return f"Document indexed successfully. Created {len(chunks)} chunks."


# -------------------------
# Summarization
# -------------------------

def summarize():

    docs = load_pdfs(DATA_DIR)

    split_docs = split_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = [d.page_content for d in split_docs]

    llm = ChatGroq(model=GROQ_CHAT_MODEL)

    if USE_TRANSFORMERS_SUMMARY:
        _, final = summarize_chunks(chunks, SUMMARIZER_MODEL)
    else:
        _, final = summarize_with_llm(llm, chunks)

    save_summary_pdf("summary.pdf", final)

    return final


# -------------------------
# Ask Question
# -------------------------

def ask(question):

    llm = ChatGroq(
        model=GROQ_CHAT_MODEL
    )

    vectorstore = get_vectorstore(
        [],
        DB_DIR,
        EMBEDDING_MODEL
    )

    docs = retrieve_chunks(
        vectorstore,
        question,
        RETRIEVE_K
    )

    answer = answer_question(
        llm,
        docs,
        question
    )[0]

    return answer


# -------------------------
# Gradio UI
# -------------------------

with gr.Blocks(
    title="PDF RAG Assistant"
) as ui:

    gr.Markdown(
        """
        # 📄 PDF RAG Assistant

        Upload your documents and chat with them using
        Retrieval Augmented Generation.
        """
    )

    with gr.Tab("Document Upload"):

        pdf_upload = gr.File(
            label="Upload PDF files",
            file_count="multiple",
            file_types=[".pdf"]
        )

        upload_status = gr.Textbox(
            label="Status"
        )

        gr.Button(
            "Upload Documents"
        ).click(
            upload_pdf,
            inputs=pdf_upload,
            outputs=upload_status
        )

        gr.Button(
            "Build Knowledge Base"
        ).click(
            build_index,
            outputs=upload_status
        )

    with gr.Tab("Chat with PDF"):

        question = gr.Textbox(
            label="Ask a question about your documents"
        )

        answer = gr.Textbox(
            label="Answer",
            lines=6
        )

        gr.Button(
            "Ask"
        ).click(
            ask,
            inputs=question,
            outputs=answer
        )

    with gr.Tab("Summary"):

       summary = gr.Textbox(
            label="Document Summary",
            lines=10
       )

       gr.Button(
            "Generate Summary"
        ).click(
            summarize,
            outputs=summary
        )


ui.launch()