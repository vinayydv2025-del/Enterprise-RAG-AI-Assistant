
from pathlib import Path
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT IMPORTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.rag_pipeline import (
    DATASETS,
    load_rag_components,
    run_rag_with_components,
)
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker


# ============================================================
# 2. GLOBAL COMPONENTS
# ============================================================

DATASET_CACHE = {}
RERANKER = None


# ============================================================
# 3. DATASET LOADING
# ============================================================

def load_dataset(dataset_name: str):
    """
    Load a dataset and its retrieval components once.
    Reuse cached components for subsequent API requests.
    """

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Choose from {list(DATASETS.keys())}"
        )

    if dataset_name not in DATASET_CACHE:
        print(
            f"Loading dataset into API cache: {dataset_name}",
            flush=True,
        )

        embedding_model, index, chunks = load_rag_components(
            dataset_name
        )

        hybrid_retriever = HybridRetriever(
            embedding_model,
            index,
            chunks,
        )

        DATASET_CACHE[dataset_name] = {
            "embedding_model": embedding_model,
            "index": index,
            "chunks": chunks,
            "hybrid_retriever": hybrid_retriever,
        }

    return DATASET_CACHE[dataset_name]


# ============================================================
# 4. APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize the default dataset and reranker at startup.
    """

    global RERANKER

    print("Starting Advanced Enterprise RAG API...", flush=True)
    print("Loading default Northstar dataset...", flush=True)

    load_dataset("northstar")

    print("Initializing Cross-Encoder reranker...", flush=True)

    RERANKER = CrossEncoderReranker()

    print(
        "Advanced RAG components loaded successfully!",
        flush=True,
    )

    yield

    DATASET_CACHE.clear()
    RERANKER = None

    print("Shutting down Advanced RAG API...", flush=True)


# ============================================================
# 5. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Advanced Enterprise RAG AI Assistant",
    description=(
        "Enterprise document question answering with hybrid "
        "retrieval, reranking, and evidence-based verification."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


# ============================================================
# 6. REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
    )

    dataset: str = Field(
        default="northstar",
        description="Dataset to search: northstar or apex",
    )


# ============================================================
# 7. SOURCE HELPERS
# ============================================================

def get_source_key(source: dict):
    """
    Create a consistent key for matching source documents
    and pages across validated and retrieved results.
    """

    return (
        str(source.get("source", "Unknown")),
        str(source.get("page", "Unknown")),
    )


def build_retrieved_source_lookup(retrieved_sources: list):
    """
    Match each source/page to its best retrieved chunk.

    If multiple chunks belong to the same page, retain the
    chunk with the highest reranker score.
    """

    lookup = {}

    for chunk in retrieved_sources:
        key = get_source_key(chunk)

        try:
            rerank_score = float(chunk.get("rerank_score", float("-inf")))
        except (TypeError, ValueError):
            rerank_score = float("-inf")

        existing = lookup.get(key)

        if existing is None:
            lookup[key] = chunk
            continue

        try:
            existing_score = float(
                existing.get("rerank_score", float("-inf"))
            )
        except (TypeError, ValueError):
            existing_score = float("-inf")

        if rerank_score > existing_score:
            lookup[key] = chunk

    return lookup


def format_score(value):
    """
    Convert a score to a rounded float.
    Return None if the score is unavailable or invalid.
    """

    if value is None:
        return None

    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def build_validated_sources(result: dict):
    """
    Return unique validated source pages with their retrieval
    and reranking scores.

    Scores are matched from retrieved chunks because validated
    source entries may contain only source and page metadata.
    """

    validated_sources = result.get(
        "validated_sources",
        result.get("sources", []),
    )

    retrieved_sources = result.get("retrieved_sources", [])

    score_lookup = build_retrieved_source_lookup(
        retrieved_sources
    )

    sources = []
    seen_sources = set()

    for item in validated_sources:
        key = get_source_key(item)

        if key in seen_sources:
            continue

        seen_sources.add(key)

        matching_chunk = score_lookup.get(key, {})

        sources.append({
            "source": item.get("source", "Unknown"),
            "page": item.get("page", "Unknown"),
            "rrf_score": format_score(
                matching_chunk.get("rrf_score")
            ),
            "rerank_score": format_score(
                matching_chunk.get("rerank_score")
            ),
        })

    return sources


def build_retrieved_sources(result: dict):
    """
    Format all retrieved chunks for API output.
    """

    formatted_sources = []

    for chunk in result.get("retrieved_sources", []):
        formatted_sources.append({
            "source": chunk.get("source", "Unknown"),
            "page": chunk.get("page", "Unknown"),
            "rrf_score": format_score(
                chunk.get("rrf_score")
            ),
            "rerank_score": format_score(
                chunk.get("rerank_score")
            ),
        })

    return formatted_sources


# ============================================================
# 8. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": (
            "Advanced Enterprise RAG AI Assistant API is running"
        ),
        "version": "2.1.0",
        "available_datasets": list(DATASETS.keys()),
        "architecture": [
            "FAISS Dense Retrieval",
            "BM25 Sparse Retrieval",
            "Reciprocal Rank Fusion",
            "Cross-Encoder Reranking",
            "Evidence-Based Answerability Verification",
            "Complete Source Page Reconstruction",
            "Ollama LLM Answer Generation",
        ],
    }


# ============================================================
# 9. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "loaded_datasets": list(DATASET_CACHE.keys()),
        "reranker_loaded": RERANKER is not None,
    }


# ============================================================
# 10. QUESTION ANSWERING ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    dataset = request.dataset.strip().lower()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if dataset not in DATASETS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid dataset '{dataset}'. "
                f"Choose from {list(DATASETS.keys())}."
            ),
        )

    if RERANKER is None:
        raise HTTPException(
            status_code=503,
            detail="RAG components are not ready.",
        )

    try:
        components = load_dataset(dataset)

        result = run_rag_with_components(
            question=question,
            dataset=dataset,
            embedding_model=components["embedding_model"],
            index=components["index"],
            chunks=components["chunks"],
            hybrid_retriever=components["hybrid_retriever"],
            reranker=RERANKER,
        )

        sources = build_validated_sources(result)
        retrieved_sources = build_retrieved_sources(result)

        return {
            "question": question,
            "dataset": dataset,
            "answer": result.get("answer", ""),
            "evidence_supported": result.get("supported", False),
            "verification_reason": result.get("reason", ""),
            "sources": sources,
            "retrieved_sources": retrieved_sources,
        }

    except FileNotFoundError as error:
        print("Dataset file error:", error, flush=True)

        raise HTTPException(
            status_code=500,
            detail="Required dataset files are missing.",
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        print("API error:", repr(error), flush=True)

        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while processing the question. "
                "Check the API terminal logs."
            ),
        ) from error

