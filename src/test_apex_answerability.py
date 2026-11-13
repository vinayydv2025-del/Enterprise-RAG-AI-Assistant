
import json

from src.answerability_checker import check_answerability


# ============================================================
# LOAD APEX CHUNKS
# ============================================================

with open(
    "data/benchmark_v2/processed/apex_chunks_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_CASES = [
    {
        "question": "What is the annual performance bonus?",
        "expected": True
    },
    {
        "question": "How much is the retirement fund contribution?",
        "expected": True
    },
    {
        "question": "What is the password length requirement?",
        "expected": True
    },
    {
        "question": "How many days of remote work are allowed per week?",
        "expected": True
    },
    {
        "question": "What is the domestic business travel expense limit?",
        "expected": True
    },
    {
        "question": "What is the daily meal expense limit?",
        "expected": True
    },
    {
        "question": "How much is the monthly internet reimbursement?",
        "expected": True
    },
    {
        "question": "Does Apex provide dental insurance?",
        "expected": False
    },
    {
        "question": "Does Apex provide biometric authentication?",
        "expected": False
    },
    {
        "question": "How many maternity leave days are provided?",
        "expected": False
    },
    {
        "question": "What is the password expiration interval?",
        "expected": False
    },
    {
        "question": "Does Apex provide an equipment allowance?",
        "expected": False
    },
]


# ============================================================
# RUN TESTS
# ============================================================

passed = 0
failed = 0

for index, test in enumerate(TEST_CASES, start=1):

    question = test["question"]
    expected = test["expected"]

    result = check_answerability(
        question,
        chunks
    )

    actual = result["supported"]

    success = actual == expected

    if success:
        passed += 1
    else:
        failed += 1

    print("\n" + "=" * 70)
    print("TEST:", index)
    print("QUESTION:", question)
    print("EXPECTED:", expected)
    print("ACTUAL:", actual)
    print("STATUS:", "PASS" if success else "FAIL")
    print("REASON:", result["reason"])

    print("\nVALIDATED EVIDENCE:")

    for item in result["evidence"]:
        print({
            "statement": item["statement"],
            "source": item["source"],
            "page": item["page"]
        })


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("APEX ANSWERABILITY TEST SUMMARY")
print("=" * 70)

print("TOTAL TESTS:", len(TEST_CASES))
print("PASSED:", passed)
print("FAILED:", failed)

accuracy = (passed / len(TEST_CASES)) * 100

print(f"TEST ACCURACY: {accuracy:.2f}%")
