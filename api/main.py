
from pathlib import Path
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from rag_pipeline import load_rag_components, generate_answer
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker
from answerability_checker import check_answerability


# Global RAG components
embedding_model = None
index = None
chunks = None
hybrid_retriever = None
reranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global embedding_model, index, chunks
    global hybrid_retriever, reranker

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    print("Initializing hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        embedding_model,
        index,
        chunks
    )

    print("Initializing Cross-Encoder reranker...")

    reranker = CrossEncoderReranker()

    print("Advanced RAG components loaded successfully!")

    yield

    print("Shutting down Advanced RAG API...")


app = FastAPI(
    title="Advanced Enterprise RAG AI Assistant",
    description=(
        "Enterprise document question answering "
        "with hybrid retrieval, reranking and "
        "evidence-based answer verification."
    ),
    version="2.0.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=1000
    )


@app.get("/")
def home():

    return {
        "message": "Advanced Enterprise RAG AI Assistant API is running",
        "version": "2.0.0",
        "architecture": [
            "FAISS Dense Retrieval",
            "BM25 Sparse Retrieval",
            "Reciprocal Rank Fusion",
            "Cross-Encoder Reranking",
            "Evidence-Based Answerability Verification",
            "Local LLM Answer Generation"
        ]
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if hybrid_retriever is None or reranker is None:

        raise HTTPException(
            status_code=503,
            detail="RAG components are not ready."
        )

    try:

        # Step 1: Hybrid retrieval
        candidates = hybrid_retriever.search(
            question,
            top_k=10
        )

        # Step 2: Cross-Encoder reranking
        reranked = reranker.rerank(
            question,
            candidates,
            top_k=5
        )

        relevant = []

        for chunk in reranked:

            relevant.append({
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"],
                "rrf_score": chunk["rrf_score"],
                "rerank_score": chunk["rerank_score"]
            })

        # Step 3: Evidence verification
        if relevant:

            verification = check_answerability(
                question,
                relevant
            )

        else:

            verification = {
                "supported": False,
                "reason": "No relevant documents were retrieved."
            }

        # Step 4: Answer generation or abstention
        if verification["supported"]:

            answer = generate_answer(
                question,
                relevant
            )

        else:

            answer = (
                "I could not find this information "
                "in the provided documents."
            )

        sources = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "rrf_score": round(chunk["rrf_score"], 4),
                "rerank_score": round(chunk["rerank_score"], 4)
            }
            for chunk in relevant
        ]

        return {
            "question": question,
            "answer": answer,
            "evidence_supported": verification["supported"],
            "verification_reason": verification["reason"],
            "sources": sources
        }

    except Exception as error:

        print("API error:", error)

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question."
        )


