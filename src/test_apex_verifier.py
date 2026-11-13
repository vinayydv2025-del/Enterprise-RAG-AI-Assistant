
from src.deterministic_verifier import (
    verify_evidence_deterministically
)



tests = [
    {
        "question": "What is the monthly internet reimbursement?",
        "evidence": "Eligible remote employees can claim internet reimbursement up to INR 800 per month.",
        "expected": False,
    },
    {
        "question": "What is the domestic travel expense limit?",
        "evidence": "Domestic business travel expenses are reimbursed up to INR 6,000 per month.",
        "expected": False,
    },
    {
        "question": "What is the maximum daily meal expense?",
        "evidence": "Meal expenses are reimbursed up to INR 800 per month.",
        "expected": False,
    },
    {
        "question": "What is the annual performance bonus?",
        "evidence": "Employees receive an annual salary increment of up to 15%.",
        "expected": False,
    },
    {
        "question": "Does Apex provide a work from home equipment allowance?",
        "evidence": "Eligible remote employees can claim internet reimbursement up to INR 1,200 per month.",
        "expected": False,
    },
    {
        "question": "What is the retirement fund contribution?",
        "evidence": "The company contributes 8% of an eligible employee's basic salary to the retirement fund every month.",
        "expected": True,
    },
]


for i, test in enumerate(tests, start=1):
    result = verify_evidence_deterministically(
        test["question"],
        test["evidence"]
    )

    actual = result.get("supported") is True

    print(
        f"TEST {i} | "
        f"EXPECTED: {test['expected']} | "
        f"ACTUAL: {actual} | "
        f"{result['reason']}"
    )
