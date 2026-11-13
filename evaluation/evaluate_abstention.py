
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_rag_components
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker


QUESTIONS = [
    # Answerable questions
    ("How many annual leave days do employees receive?", True),
    ("How much sick leave can employees take?", True),
    ("What is the minimum password length?", True),
    ("When must employees report security incidents?", True),
    ("How many days per week can employees work remotely?", True),

    # Unsupported questions
    ("How many maternity leave days are provided?", False),
    ("What is the employee salary increment percentage?", False),
    ("Which bank does the company use for salary deposits?", False),
    ("Can unused annual leave be carried forward?", False),
    ("What is the employee notice period?", False),
    ("How many paternity leave days are provided?", False),
    ("What is the company's travel reimbursement limit?", False),
    ("Which hospitals are covered by employee insurance?", False),
    ("Does the company reimburse home internet expenses?", False),
    ("What is the employee bonus structure?", False)
]


def main():

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    hybrid = HybridRetriever(
        embedding_model,
        index,
        chunks
    )

    reranker = CrossEncoderReranker()

    print("\n" + "=" * 75)
    print("ABSTENTION CONFIDENCE EVALUATION")
    print("=" * 75)

    for question, answerable in QUESTIONS:

        candidates = hybrid.search(
            question,
            top_k=10
        )

        results = reranker.rerank(
            question,
            candidates,
            top_k=5
        )

        if results:
            top_score = results[0]["rerank_score"]
            second_score = (
                results[1]["rerank_score"]
                if len(results) > 1 else None
            )
        else:
            top_score = None
            second_score = None

        print(f"\nQuestion: {question}")
        print(f"Expected answerable: {answerable}")
        print(f"Top reranker score: {top_score}")
        print(f"Second reranker score: {second_score}")

        if results:
            print(
                "Top retrieved chunk:",
                results[0]["chunk_id"]
            )

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
