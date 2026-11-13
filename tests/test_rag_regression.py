
import sys
import re
from pathlib import Path

import pytest


# ============================================================
# 1. PROJECT IMPORT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import run_rag


# ============================================================
# 2. ANSWER NORMALIZATION
# ============================================================

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}


def normalize_answer(text):
    """
    Normalize answer text so equivalent number formats
    can be compared.

    Examples:
        six months -> 6 months
        6 months -> 6 months
        two days -> 2 days
        Multi-Factor Authentication -> multi factor authentication
    """

    text = str(text).lower()

    # Normalize written number words.
    for word, digit in NUMBER_WORDS.items():
        text = re.sub(
            rf"\b{word}\b",
            digit,
            text,
        )

    # Normalize hyphens and punctuation.
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# 3. REGRESSION TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # APEX DATASET
    # --------------------------------------------------------

    {
        "name": "Apex confidential information access",
        "dataset": "apex",
        "question": "Who is allowed to access confidential information?",
        "expected_supported": True,
        "expected_source": "Apex_IT_Security_Policy.pdf",
        "expected_answer": "business authorization",
    },

    {
        "name": "Apex external contractor access",
        "dataset": "apex",
        "question": "Can external contractors access confidential information?",
        "expected_supported": False,
        "expected_source": None,
        "expected_answer": None,
    },

    # --------------------------------------------------------
    # NORTHSTAR DATASET
    # --------------------------------------------------------

    {
        "name": "Northstar annual leave",
        "dataset": "northstar",
        "question": "How many annual leave days do employees receive?",
        "expected_supported": True,
        "expected_source": "Leave_and_Attendance_Policy.pdf",
        "expected_answer": "20",
    },

    {
        "name": "Northstar sick leave",
        "dataset": "northstar",
        "question": "How many sick leave days are provided?",
        "expected_supported": True,
        "expected_source": "Leave_and_Attendance_Policy.pdf",
        "expected_answer": "10",
    },

    {
        "name": "Northstar planned leave notification",
        "dataset": "northstar",
        "question": "How many working days in advance must planned leave be requested?",
        "expected_supported": True,
        "expected_source": "Leave_and_Attendance_Policy.pdf",
        "expected_answer": "5 working days",
    },

    {
        "name": "Northstar late arrival notification",
        "dataset": "northstar",
        "question": "What should employees do if they arrive more than 15 minutes late?",
        "expected_supported": True,
        "expected_source": "Leave_and_Attendance_Policy.pdf",
        "expected_answer": "notify",
    },

    {
        "name": "Northstar minimum password length",
        "dataset": "northstar",
        "question": "What is the minimum password length required?",
        "expected_supported": True,
        "expected_source": "Information_Security_Policy.pdf",
        "expected_answer": "12 characters",
    },

    {
        "name": "Northstar security incident reporting",
        "dataset": "northstar",
        "question": "Within how much time must employees report security incidents?",
        "expected_supported": True,
        "expected_source": "Information_Security_Policy.pdf",
        "expected_answer": "one hour",
    },

    {
        "name": "Northstar multi-factor authentication",
        "dataset": "northstar",
        "question": "Is multi-factor authentication mandatory?",
        "expected_supported": True,
        "expected_source": "Information_Security_Policy.pdf",
        "expected_answer": "Multi-Factor Authentication",
    },

    {
        "name": "Northstar laptop security",
        "dataset": "northstar",
        "question": "What security measures are required for company laptops?",
        "expected_supported": True,
        "expected_source": "Information_Security_Policy.pdf",
        "expected_answer": "encryption",
    },

    {
        "name": "Northstar remote work",
        "dataset": "northstar",
        "question": "How many days per week can employees work remotely?",
        "expected_supported": True,
        "expected_source": "Employee_HR_Policy.pdf",
        "expected_answer": "two days",
    },

    {
        "name": "Northstar probation period",
        "dataset": "northstar",
        "question": "What is the employee probation period?",
        "expected_supported": True,
        "expected_source": "Employee_HR_Policy.pdf",
        "expected_answer": "6 months",
    },

    {
        "name": "Northstar performance reviews",
        "dataset": "northstar",
        "question": "How frequently are employee performance reviews conducted?",
        "expected_supported": True,
        "expected_source": "Employee_HR_Policy.pdf",
        "expected_answer": "six months",
    },

    {
        "name": "Northstar unsupported maternity leave",
        "dataset": "northstar",
        "question": "How many weeks of maternity leave do employees receive?",
        "expected_supported": False,
        "expected_source": None,
        "expected_answer": None,
    },

]


# ============================================================
# 4. PARAMETERIZED FULL-PIPELINE REGRESSION TEST
# ============================================================

@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=[case["name"] for case in TEST_CASES],
)
def test_rag_regression(case):

    print("\n" + "=" * 70)
    print(f"TEST: {case['name']}")
    print(f"DATASET: {case['dataset']}")
    print(f"QUESTION: {case['question']}")
    print("=" * 70)

    # --------------------------------------------------------
    # Run the complete RAG pipeline
    # --------------------------------------------------------

    result = run_rag(
        question=case["question"],
        dataset=case["dataset"],
    )

    # --------------------------------------------------------
    # Validate result structure
    # --------------------------------------------------------

    assert isinstance(result, dict), (
        "RAG pipeline must return a dictionary."
    )

    required_fields = [
        "answer",
        "supported",
        "sources",
        "reason",
    ]

    for field in required_fields:
        assert field in result, (
            f"Missing required result field: {field}"
        )

    answer = str(result["answer"])
    sources = result["sources"]

    # --------------------------------------------------------
    # Validate answerability decision
    # --------------------------------------------------------

    assert result["supported"] is case["expected_supported"], (
        f"Incorrect answerability decision.\n"
        f"Expected: {case['expected_supported']}\n"
        f"Actual: {result['supported']}\n"
        f"Answer: {answer}\n"
        f"Reason: {result['reason']}"
    )

    # --------------------------------------------------------
    # Supported questions
    # --------------------------------------------------------

    if case["expected_supported"]:

        assert isinstance(sources, list), (
            "Sources must be returned as a list."
        )

        assert len(sources) > 0, (
            "Supported answers must contain validated sources."
        )

        # Verify expected document citation.
        source_found = any(
            case["expected_source"] in str(source)
            for source in sources
        )

        assert source_found, (
            f"Expected source not found: "
            f"{case['expected_source']}\n"
            f"Actual sources: {sources}"
        )

        # Verify expected answer information.
        normalized_answer = normalize_answer(answer)

        normalized_expected = normalize_answer(
            case["expected_answer"]
        )

        assert normalized_expected in normalized_answer, (
            f"Expected answer information was not found.\n"
            f"Expected: {case['expected_answer']}\n"
            f"Normalized expected: {normalized_expected}\n"
            f"Actual answer: {answer}\n"
            f"Normalized answer: {normalized_answer}"
        )

    # --------------------------------------------------------
    # Unsupported questions
    # --------------------------------------------------------

    else:

        assert sources == [], (
            "Unsupported questions must not return validated sources."
        )

        answer_lower = answer.lower()

        abstention_phrases = [
            "could not find",
            "not found",
            "unable to find",
            "not available",
            "insufficient evidence",
        ]

        assert any(
            phrase in answer_lower
            for phrase in abstention_phrases
        ), (
            "Unsupported question did not produce an appropriate "
            "abstention response.\n"
            f"Actual answer: {answer}"
        )

    # --------------------------------------------------------
    # Test summary
    # --------------------------------------------------------

    print("\nTEST PASSED")
    print(f"Answer: {answer}")
    print(f"Supported: {result['supported']}")
    print(f"Sources: {sources}")
    print(f"Reason: {result['reason']}")

