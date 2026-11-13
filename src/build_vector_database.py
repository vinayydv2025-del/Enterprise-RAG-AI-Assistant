
from src.generate_embeddings import generate_embeddings

import faiss
import numpy as np
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_VECTOR_DIR = BASE_DIR / "vectorstore"


def build_vector_database(data_dir=None, vector_dir=DEFAULT_VECTOR_DIR):

    vector_dir = Path(vector_dir)

    if not vector_dir.is_absolute():
        vector_dir = BASE_DIR / vector_dir

    vector_dir.mkdir(parents=True, exist_ok=True)

    chunks, embeddings = generate_embeddings(data_dir)

    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Invalid or empty embeddings array.")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks does not match number of embeddings."
        )

    dimension = embeddings.shape[1]

    # Inner product works as cosine similarity
    # because embeddings are normalized.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    # Save FAISS index
    index_path = vector_dir / "apex_faiss.index"

    faiss.write_index(
        index,
        str(index_path)
    )

    # Save corresponding text and metadata
    metadata_path = vector_dir / "apex_chunks_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 50)
    print("VECTOR DATABASE SUMMARY")
    print("=" * 50)

    print("Total vectors stored:", index.ntotal)
    print("Vector dimensions:", dimension)
    print("Index type:", type(index).__name__)
    print("FAISS index:", index_path)
    print("Metadata:", metadata_path)

    print("\nVector database created successfully!")

    return index, chunks


if __name__ == "__main__":

    build_vector_database()
