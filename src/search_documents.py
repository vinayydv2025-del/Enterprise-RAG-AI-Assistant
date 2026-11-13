
from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "vectorstore"

def search_documents(query, top_k=3):

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(
        str(VECTOR_DIR / "enterprise_faiss.index")
    )

    with open(
        VECTOR_DIR / "chunks_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(query_embedding, top_k)

    print("\n" + "=" * 60)
    print("USER QUESTION:", query)
    print("=" * 60)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        chunk = chunks[idx]

        print(f"\nRESULT {rank}")
        print("Similarity Score:", round(float(score), 4))
        print("Source:", chunk["metadata"]["source"])
        print("Page:", chunk["metadata"]["page"])
        print("Chunk:", chunk["metadata"]["chunk_number"])
        print("Content:", chunk["text"])

if __name__ == "__main__":

    question = "What is the probation period?"

    search_documents(question)
