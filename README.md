
# Enterprise RAG-Based AI Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from company policy documents.

The application uses a local Large Language Model (LLM), semantic search, and a vector database to generate context-grounded answers with document citations.

## Features

- PDF document ingestion
- Text extraction and chunking
- Semantic embeddings using Sentence Transformers
- FAISS vector database for similarity search
- Local LLM inference using Ollama and Llama 3.2
- Similarity threshold-based retrieval filtering
- Source-aware answers
- FastAPI REST API
- Interactive Streamlit interface
- Automated API testing with Pytest
- Basic RAG evaluation framework

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| LLM | Llama 3.2 3B |
| Local LLM Runtime | Ollama |
| Backend | FastAPI |
| Frontend | Streamlit |
| PDF Processing | PyPDF |
| Testing | Pytest |

## Project Structure

```text
Enterprise RAG AI Assistant/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
│   ├── test_questions.json
│   └── run_evaluation.py
│
├── src/
│   ├── create_sample_documents.py
│   ├── document_ingestion.py
│   ├── text_chunking.py
│   ├── generate_embeddings.py
│   ├── build_vector_database.py
│   ├── search_documents.py
│   └── rag_pipeline.py
│
├── tests/
│   └── test_api.py
│
├── vectorstore/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Enterprise RAG AI Assistant"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and start Ollama

Install Ollama and download the local model:

```bash
ollama pull llama3.2:3b
```

### 5. Generate the sample documents

```bash
python src/create_sample_documents.py
```

### 6. Build the vector database

Run:

```bash
python src/build_vector_database.py
```

This generates the FAISS index and chunk metadata.

## Running the Application

### Start the FastAPI backend

```bash
uvicorn api.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

### Start the Streamlit frontend

Open another terminal:

```bash
streamlit run app/streamlit_app.py
```

Open:

http://localhost:8501

## Evaluation

Run the RAG evaluation script:

```bash
python evaluation/run_evaluation.py
```

The initial evaluation achieved:

- Test questions: 5
- Passed: 5
- Failed: 0
- Pass rate: 100%

Note: This result applies to the initial five-question evaluation dataset.

## API Testing

Run:

```bash
python -m pytest tests/test_api.py -v
```

Initial API test results:

- Home endpoint: PASS
- Ask endpoint: PASS
- Invalid request validation: PASS

## How It Works

1. PDF documents are loaded and text is extracted.
2. Extracted text is divided into smaller chunks.
3. Sentence Transformers generates numerical embeddings.
4. FAISS stores the embeddings for similarity search.
5. A user submits a question.
6. The question is embedded and relevant document chunks are retrieved.
7. A similarity threshold filters irrelevant results.
8. Ollama runs Llama 3.2 locally to generate a grounded answer.
9. The answer and source information are returned through FastAPI and displayed in Streamlit.

## Limitations and Future Improvements

- Expand the evaluation dataset.
- Improve retrieval quality and chunking strategies.
- Add more realistic enterprise documents.
- Implement advanced RAG evaluation.
- Add Docker deployment.
- Improve API error handling and logging.
- Explore hybrid search and reranking.

## Cost

The project is designed to run locally using open-source tools and does not require paid LLM API access.
