
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from src.dense_retriever import retrieve_documents


BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = BASE_DIR / "data" / "benchmark_v2" / "processed"

INDEX_PATH = VECTOR_DIR / "apex_faiss.index"

METADATA_PATH = VECTOR_DIR / "apex_chunks_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():

    print("Loading Apex FAISS index...")

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(MODEL_NAME)

    questions = [
        "What is the annual performance bonus limit?",
        "How quickly must security incidents be reported?",
        "What is the monthly internet reimbursement limit?"
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        results = retrieve_documents(
            question=question,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
            top_k=3
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Similarity score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
