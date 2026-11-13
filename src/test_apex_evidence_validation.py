
import json

from src.answerability_checker import validate_llm_evidence


# Load the actual Apex document chunks.
with open(
    "data/benchmark_v2/processed/apex_chunks_metadata.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)


# Select an actual chunk.
chunk = chunks[0]

evidence = [
    {
        "statement": "Apex Technologies provides unlimited internet reimbursement of INR 50,000 per month."
    }
]

# Validate the evidence.
result = validate_llm_evidence(
    evidence,
    [
        {
            "chunk_id": 0,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "score": 0.95
        }
    ]
)

print("VALIDATED EVIDENCE:")
print(result)

print("\nSOURCE:", result[0]["source"] if result else "FAILED")
print("PAGE:", result[0]["page"] if result else "FAILED")
