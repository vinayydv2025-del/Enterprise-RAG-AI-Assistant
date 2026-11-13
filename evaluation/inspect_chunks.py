import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

metadata_file = BASE_DIR / "vectorstore" / "chunks_metadata.json"

with open(metadata_file, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("=" * 70)
print("DOCUMENT CHUNK INSPECTION")
print("=" * 70)

for i, chunk in enumerate(chunks):
    print(f"\nCHUNK ID: {i}")
    print(f"SOURCE: {chunk['metadata']['source']}")
    print(f"PAGE: {chunk['metadata']['page']}")
    print(f"TEXT: {chunk['text']}")
    print("-" * 70)