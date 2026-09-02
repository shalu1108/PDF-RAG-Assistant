# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) app that lets you upload PDF documents,
ask questions grounded in their content, and generate summaries — served through
a Gradio web UI.

## Tech Stack
- **LangChain** – orchestration (document loading, splitting, retrieval)
- **Chroma** – vector database for storing document embeddings
- **HuggingFace `sentence-transformers/all-MiniLM-L6-v2`** – embedding model
- **Groq (`ChatGroq`)** – LLM for question answering and summarization
- **Transformers (`flan-t5-base`)** – optional local summarization fallback
- **Gradio** – web interface

## Setup

1. Clone the repo and enter the folder:
   ```bash
   git clone <your-repo-url>
   cd pdf-rag-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

5. Run the app:
   ```bash
   python app.py
   ```

6. Open the URL Gradio prints in the terminal (usually `http://127.0.0.1:7860`).

## How to use
1. **Document Upload tab** – upload one or more PDFs, then click "Build Knowledge Base" to chunk and index them into Chroma.
2. **Chat with PDF tab** – ask questions; answers are grounded in retrieved chunks, and the model responds "I don't know" if the answer isn't in the documents.
3. **Summary tab** – generate a combined summary of the uploaded document(s), also saved as `summary.pdf`.

## Project Structure
```
├── app.py                  # Gradio UI and entry point
├── config.py                # Chunking, embedding, and model settings
├── ingestion/                # PDF loading, text cleaning, chunking, vector store
├── rag/                      # Retrieval and question answering
├── summarization/            # LLM-based and local (Transformers) summarization
├── utils/                    # Prompts and PDF export
└── requirements.txt
```

## Notes
- Only PDF files are supported.
- Summarization defaults to the Groq LLM; set `USE_TRANSFORMERS_SUMMARY = True` in `config.py` to use the local `flan-t5` model instead.
