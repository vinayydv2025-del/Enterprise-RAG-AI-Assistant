
import json
from pathlib import Path

from src.bm25_retriever import BM25Retriever


BASE_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    BASE_DIR
    / "data"
    / "benchmark_v2"
    / "processed"
    / "apex_chunks_metadata.json"
)


def main():

    print("Loading Apex document chunks...")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Total chunks loaded:", len(chunks))

    print("\nBuilding BM25 index...")

    bm25_retriever = BM25Retriever(chunks)

    questions = [
        "What is the annual performance bonus limit?",
        "How quickly must security incidents be reported?",
        "What is the monthly internet reimbursement limit?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        results = bm25_retriever.search(
            query=question,
            top_k=3
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"BM25 score: {result['score']:.4f}")
            print(f"Source: {result['metadata']['source']}")
            print(f"Page: {result['metadata']['page']}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
