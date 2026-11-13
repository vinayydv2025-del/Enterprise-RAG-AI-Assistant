
import sys
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import load_rag_components
from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker
from answerability_checker import check_answerability


QUESTIONS = [
    # Existing benchmark questions
    ("How many annual leave days do employees receive?", True),
    ("How much sick leave can employees take?", True),
    ("What is the minimum password length?", True),
    ("When must employees report security incidents?", True),
    ("How many days per week can employees work remotely?", True),
    ("How many maternity leave days are provided?", False),
    ("What is the employee salary increment percentage?", False),
    ("Which bank does the company use for salary deposits?", False),
    ("Can unused annual leave be carried forward?", False),
    ("What is the employee notice period?", False),
    ("How many paternity leave days are provided?", False),
    ("What is the company's travel reimbursement limit?", False),
    ("Which hospitals are covered by employee insurance?", False),
    ("Does the company reimburse home internet expenses?", False),
    ("What is the employee bonus structure?", False),
    ("At what time does the workday begin?", True),
    ("When does the working day end?", True),
    ("How long is the employee probation period?", True),
    ("How frequently are performance reviews conducted?", True),
    ("Is multi-factor authentication compulsory?", True),
    ("What access control principle does the company follow?", True),
    ("Are company laptops required to have full-disk encryption?", True),
    ("What should happen automatically when an employee leaves their screen?", True),
    ("How many public holidays are provided annually?", True),
    ("What should employees do if they arrive more than 15 minutes late?", True),
    ("Can employees work remotely five days a week?", True),
    ("Does the company provide paid parental leave?", False),
    ("What is the annual salary of a software engineer?", False),
    ("Does the company offer stock options?", False),
    ("What is the retirement age at the company?", False),
    ("Can employees use personal email to share confidential company files?", True),
    ("Does the company provide relocation allowances?", False),
    ("What is the company's maternity insurance coverage limit?", False),
    ("How many casual leave days are available?", False),
    ("Does the company provide a work-from-home equipment allowance?", False),

    # Additional paraphrases
    ("What is the yearly entitlement for paid vacation?", True),
    ("How many paid sick days are available each year?", True),
    ("How many characters must a password contain at minimum?", True),
    ("How quickly must a security incident be reported?", True),
    ("What is the maximum weekly remote working allowance?", True),
    ("When does the company expect employees to start work?", True),
    ("What time should employees finish their workday?", True),
    ("What is the duration of the probationary period?", True),
    ("How often does management review employee performance?", True),
    ("Is MFA required for internal business applications?", True),
    ("Which access-control model follows minimum necessary permissions?", True),
    ("Must laptops encrypt their entire storage drives?", True),
    ("Does the laptop automatically lock its screen?", True),
    ("How many official holidays does the company observe per year?", True),
    ("Who should be informed when an employee is over 15 minutes late?", True),

    # Negative policy questions
    ("Is remote work allowed without manager approval?", True),
    ("Can employees share confidential information using personal email?", True),
    ("Are employees allowed to skip multi-factor authentication?", True),
    ("Can an employee access information unrelated to their job?", True),
    ("Is a password with only eight characters compliant?", True),
    ("Can security incidents be reported after several days?", True),
    ("Are company laptops exempt from full-disk encryption?", True),
    ("Can employees work remotely more than two days per week?", True),
    ("Is screen locking unnecessary on company laptops?", True),
    ("Can employees arrive more than 15 minutes late without notifying their manager?", True),
    ("Does the policy allow unlimited remote working?", True),
    ("Can confidential company files be sent to a personal email account?", True),
    ("Is MFA optional for company cloud platforms?", True),
    ("Can employees access all company information regardless of their role?", True),
    ("Does the policy permit passwords shorter than 12 characters?", True),

    # Numerical limits and boundaries
    ("Is two days per week within the remote work limit?", True),
    ("Is three days of remote work per week permitted?", True),
    ("Does the company require at least 12 password characters?", True),
    ("Is a 12-character password long enough to meet the minimum?", True),
    ("Does the policy specify a one-hour security incident reporting deadline?", True),
    ("Is reporting a security incident within 30 minutes compliant?", True),
    ("Does the company provide exactly 20 annual leave days?", True),
    ("Are 10 sick leave days provided annually?", True),
    ("Does the company observe 12 public holidays?", True),
    ("Is a five-month probation period specified?", True),
    ("Does probation last six months?", True),
    ("Are performance reviews conducted every six months?", True),
    ("Does the policy mention a 24-hour incident reporting deadline?", True),
    ("Are employees required to notify their manager after 15 minutes of lateness?", True),
    ("Does the company permit remote work for five days every week?", True),

    # Multi-part questions
    ("How many annual leave and sick leave days are provided?", True),
    ("What are the working hours and which days are employees expected to work?", True),
    ("What password length and character types are required?", True),
    ("Which platforms require MFA and what access control principle is followed?", True),
    ("What are the probation duration and performance review frequency?", True),
    ("How many public holidays are provided and how early must planned leave be requested?", True),
    ("What are the annual leave and public holiday entitlements?", True),
    ("What encryption and screen-lock protections are required for laptops?", True),
    ("How many remote days are permitted and is manager approval required?", True),
    ("What should employees do during emergency leave and when arriving late?", True),

    # Similar but distinct concepts
    ("Is least privilege the same as multi-factor authentication?", False),
    ("Does the policy specify biometric authentication requirements?", False),
    ("Does the company require VPN usage for remote work?", False),
    ("Does the policy specify password expiration intervals?", False),
    ("Are employees required to use hardware security keys?", False),
    ("Does the company provide dental insurance?", False),
    ("Does the company specify a retirement contribution percentage?", False),
    ("Are employees given a company phone allowance?", False),
    ("Does the company specify a maximum number of consecutive leave days?", False),
    ("Does the policy specify a laptop replacement schedule?", False),

    # Additional unsupported questions
    ("What is the company's annual revenue?", False),
    ("Who is the company's chief executive officer?", False),
    ("How many employees work at the company?", False),
    ("What is the company's registered office address?", False),
    ("Which payroll software does the company use?", False),
    ("What is the annual training budget?", False),
    ("Does the company offer stock purchase plans?", False),
    ("What is the notice period during probation?", False),
    ("How many work-from-home internet reimbursements are allowed?", False),
    ("What is the company's parental insurance limit?", False),
]


def calculate_metrics(tp, tn, fp, fn):

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-10)
    )

    unsupported_rejection_rate = tn / max(tn + fp, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "unsupported_rejection_rate": unsupported_rejection_rate
    }


def main():

    print("Loading RAG components...")

    embedding_model, index, chunks = load_rag_components()

    hybrid = HybridRetriever(
        embedding_model,
        index,
        chunks
    )

    reranker = CrossEncoderReranker()

    tp = tn = fp = fn = 0
    results = []

    print("\nTotal benchmark questions:", len(QUESTIONS))

    for number, (question, expected) in enumerate(QUESTIONS, start=1):

        candidates = hybrid.search(
            question,
            top_k=10
        )

        reranked = reranker.rerank(
            question,
            candidates,
            top_k=5
        )

        retrieved_chunks = [
            {
                "text": chunk["text"],
                "source": chunk["metadata"]["source"],
                "page": chunk["metadata"]["page"]
            }
            for chunk in reranked
        ]

        result = check_answerability(
            question,
            retrieved_chunks
        )

        predicted = result["supported"]
        correct = predicted == expected

        if expected and predicted:
            tp += 1
        elif not expected and not predicted:
            tn += 1
        elif not expected and predicted:
            fp += 1
        else:
            fn += 1

        results.append({
"question_id": number,
"question": question,
"expected_supported": expected,
"predicted_supported": predicted,
"correct": correct,

"reason": result["reason"],

# Stage 1: Retrieved document chunks
"retrieved_chunks": retrieved_chunks,

# Stage 2: Extracted evidence
"extracted_evidence": result.get("evidence", []),

# Retrieved source filenames
"sources": [
    chunk["source"]
    for chunk in retrieved_chunks
]
})


        print(
            f"[{number}/{len(QUESTIONS)}] "
            f"{'PASS' if correct else 'FAIL'} - {question}"
        )

    metrics = calculate_metrics(tp, tn, fp, fn)

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(QUESTIONS),
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        },
        "metrics": metrics,
        "results": results
    }

    output_dir = BASE_DIR / "evaluation"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "answerability_evaluation_report.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("ANSWERABILITY EVALUATION SUMMARY")
    print("=" * 65)

    print("Total questions:", len(QUESTIONS))
    print("True Positives:", tp)
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    print("\nAccuracy:", f"{metrics['accuracy'] * 100:.2f}%")
    print("Precision:", f"{metrics['precision'] * 100:.2f}%")
    print("Recall:", f"{metrics['recall'] * 100:.2f}%")
    print("F1-score:", f"{metrics['f1_score'] * 100:.2f}%")
    print(
        "Unsupported rejection rate:",
        f"{metrics['unsupported_rejection_rate'] * 100:.2f}%"
    )

    print("\nDetailed report saved to:")
    print(output_file)


if __name__ == "__main__":
    main()


