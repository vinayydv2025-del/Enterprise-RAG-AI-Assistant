
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker


BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = BASE_DIR / "data" / "benchmark_v2" / "processed"

INDEX_PATH = VECTOR_DIR / "apex_faiss.index"

METADATA_PATH = VECTOR_DIR / "apex_chunks_metadata.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def main():

    print("Loading Apex FAISS index...")

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("\nBuilding hybrid retriever...")

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        index=index,
        chunks=chunks
    )

    print("\nLoading Cross-Encoder reranker...")

    reranker = CrossEncoderReranker()

    questions = [
        "What is the annual performance bonus limit?",
        "How quickly must security incidents be reported?",
        "What is the monthly internet reimbursement limit?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        # Retrieve 10 candidates using hybrid retrieval
        hybrid_results = hybrid_retriever.search(
            query=question,
            top_k=10
        )

        # Rerank candidates using Cross-Encoder
        reranked_results = reranker.rerank(
            query=question,
            retrieved_chunks=hybrid_results,
            top_k=3
        )

        for rank, result in enumerate(reranked_results, start=1):

            print(f"\nRank: {rank}")
            print(f"Rerank score: {result['rerank_score']:.4f}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Page: {result['metadata']['page']}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
