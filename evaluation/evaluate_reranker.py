
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_rag_components
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker

BENCHMARK_FILE = BASE_DIR / "evaluation" / "retrieval_benchmark.json"

TOP_K = 5


def main():

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
        benchmark = json.load(file)

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    hybrid = HybridRetriever(embedding_model, index, chunks)
    reranker = CrossEncoderReranker()

    total = 0
    hits = 0
    reciprocal_ranks = []

    print("\n" + "=" * 70)
    print("CROSS-ENCODER RERANKING EVALUATION")
    print("=" * 70)

    for item in benchmark:

        if not item["answerable"]:
            continue

        total += 1

        query = item["question"]
        expected_ids = item["expected_chunk_ids"]

        candidates = hybrid.search(query, top_k=TOP_K)

        results = reranker.rerank(
            query,
            candidates,
            top_k=TOP_K
        )

        retrieved_ids = [
            result["chunk_id"] for result in results
        ]

        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
            if chunk_id in expected_ids
        ]

        if relevant_ranks:
            hits += 1
            reciprocal_ranks.append(1 / relevant_ranks[0])
        else:
            reciprocal_ranks.append(0)

        print(f"\nQuestion: {query}")
        print(f"Expected: {expected_ids}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Hit: {bool(relevant_ranks)}")

    print("\n" + "=" * 70)
    print("RERANKER SUMMARY")
    print("=" * 70)

    print(f"Total Questions: {total}")
    print(f"Hit Rate@5: {hits / total:.4f}")
    print(f"MRR@5: {sum(reciprocal_ranks) / total:.4f}")


if __name__ == "__main__":
    main()
