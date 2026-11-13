
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_rag_components
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker


def main():

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    hybrid_retriever = HybridRetriever(
        embedding_model,
        index,
        chunks
    )

    query = "What is the minimum password length?"

    print("\nRetrieving initial candidates...")

    candidates = hybrid_retriever.search(
        query,
        top_k=5
    )

    print("\nLoading Cross-Encoder...")

    reranker = CrossEncoderReranker()

    print("\nReranking candidates...")

    results = reranker.rerank(
        query,
        candidates,
        top_k=5
    )

    print("\nCROSS-ENCODER RERANKING RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Text: {result['text'][:250]}")


if __name__ == "__main__":
    main()
