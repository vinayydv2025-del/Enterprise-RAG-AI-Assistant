
from pathlib import Path
import json
import sys
import re

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from rag_pipeline import (
    load_rag_components,
    retrieve_documents,
    generate_answer
)

EVALUATION_FILE = BASE_DIR / "evaluation" / "test_questions.json"
SIMILARITY_THRESHOLD = 0.45

STOPWORDS = {"and", "or", "of", "the", "with", "per", "a", "an"}


def check_answer(expected, actual):
    expected_words = set(
        re.findall(r"\b\w+\b", expected.lower())
    ) - STOPWORDS

    actual_words = set(
        re.findall(r"\b\w+\b", actual.lower())
    )

    return expected_words.issubset(actual_words)


def run_evaluation():

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    total = len(test_cases)
    passed = 0

    print("\n" + "=" * 60)
    print("ENTERPRISE RAG - EVALUATION REPORT")
    print("=" * 60)

    for number, test in enumerate(test_cases, start=1):

        question = test["question"]

        retrieved = retrieve_documents(
            question,
            embedding_model,
            index,
            chunks
        )

        relevant = [
            chunk for chunk in retrieved
            if chunk["score"] >= SIMILARITY_THRESHOLD
        ]

        should_answer = test["should_answer"]

        print(f"\nTEST {number}: {question}")

        if should_answer:

            source_found = any(
                chunk["source"] == test["expected_source"]
                for chunk in relevant
            )

            if relevant:
                answer = generate_answer(question, relevant)
                answer_correct = check_answer(
                    test["expected_answer"],
                    answer
                )
            else:
                answer = "No relevant information found."
                answer_correct = False

            test_passed = source_found and answer_correct

            print("Generated answer:", answer)
            print("Expected source found:", source_found)
            print("Answer keywords correct:", answer_correct)

        else:

            test_passed = len(relevant) == 0

            print("Generated answer: No sufficiently relevant information.")
            print("LLM skipped:", test_passed)

        if test_passed:
            passed += 1
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", total - passed)
    print("Pass rate:", round((passed / total) * 100, 2), "%")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    run_evaluation()
