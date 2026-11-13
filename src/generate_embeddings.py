
from src.text_chunking import split_documents
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings(data_dir=None):

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    if data_dir is None:
        chunks = split_documents()
    else:
        chunks = split_documents(data_dir)

    if not chunks:
        raise ValueError("No text chunks available for embedding.")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 50)

    print("Total text chunks:", len(texts))
    print("Embedding shape:", embeddings.shape)
    print("Vector dimensions:", embeddings.shape[1])
    print("Data type:", embeddings.dtype)

    print("\nFirst chunk:")
    print(texts[0])

    print("\nFirst 10 embedding values:")
    print(embeddings[0][:10])

    print("\nEmbedding generation completed successfully!")

    return chunks, embeddings


if __name__ == "__main__":

    chunks, embeddings = generate_embeddings()
