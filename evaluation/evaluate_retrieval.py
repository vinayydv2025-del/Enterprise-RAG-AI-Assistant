
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_rag_components, retrieve_documents

BENCHMARK_FILE = BASE_DIR / "evaluation" / "retrieval_benchmark.json"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.45


def calculate_metrics(retrieved_ids, expected_ids, k):
    retrieved = retrieved_ids[:k]

    relevant_retrieved = [
        chunk_id for chunk_id in retrieved
        if chunk_id in expected_ids
    ]

    precision = len(relevant_retrieved) / k

    recall = (
        len(set(relevant_retrieved)) / len(expected_ids)
        if expected_ids else 0
    )

    reciprocal_rank = 0

    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_ids:
            reciprocal_rank = 1 / position
            break

    hit_rate = 1 if relevant_retrieved else 0

    return precision, recall, reciprocal_rank, hit_rate


def evaluate_retrieval():

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
        benchmark = json.load(file)

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    precisions = []
    recalls = []
    reciprocal_ranks = []
    hit_rates = []

    unsupported_total = 0
    unsupported_correct = 0

    print("\n" + "=" * 70)
    print("CHUNK-LEVEL RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    for item in benchmark:

        question = item["question"]
        expected_ids = item["expected_chunk_ids"]
        answerable = item["answerable"]

        results = retrieve_documents(
            question,
            embedding_model,
            index,
            chunks,
            top_k=TOP_K
        )

        retrieved_ids = []

        for result in results:

            matching_id = -1

            for chunk_id, chunk in enumerate(chunks):

                if (
                    result["text"] == chunk["text"]
                    and result["source"] == chunk["metadata"]["source"]
                    and result["page"] == chunk["metadata"]["page"]
                ):
                    matching_id = chunk_id
                    break

            retrieved_ids.append(matching_id)

        if answerable:

            precision, recall, mrr, hit_rate = calculate_metrics(
                retrieved_ids,
                expected_ids,
                TOP_K
            )

            precisions.append(precision)
            recalls.append(recall)
            reciprocal_ranks.append(mrr)
            hit_rates.append(hit_rate)

            print(f"\nQuestion: {question}")
            print(f"Expected chunk IDs: {expected_ids}")
            print(f"Retrieved chunk IDs: {retrieved_ids}")
            print(f"Precision@{TOP_K}: {precision:.4f}")
            print(f"Recall@{TOP_K}: {recall:.4f}")
            print(f"MRR: {mrr:.4f}")
            print(f"Hit Rate: {hit_rate:.4f}")

        else:

            unsupported_total += 1

            confident_results = [
                result for result in results
                if result["score"] >= SIMILARITY_THRESHOLD
            ]

            rejected = len(confident_results) == 0

            if rejected:
                unsupported_correct += 1

            print(f"\nQuestion: {question}")
            print("Expected: No relevant document")
            print("Correctly rejected:", rejected)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    if precisions:
        print(f"Mean Precision@{TOP_K}: {sum(precisions) / len(precisions):.4f}")
        print(f"Mean Recall@{TOP_K}: {sum(recalls) / len(recalls):.4f}")
        print(f"Mean MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")
        print(f"Mean Hit Rate: {sum(hit_rates) / len(hit_rates):.4f}")
    else:
        print("No answerable questions were evaluated.")

    if unsupported_total:
        print(
            "Unsupported Rejection Rate:",
            f"{unsupported_correct / unsupported_total:.4f}"
        )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate_retrieval()
