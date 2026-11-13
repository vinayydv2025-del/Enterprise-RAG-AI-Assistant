import faiss
import numpy as np


def retrieve_documents(
    question,
    embedding_model,
    index,
    chunks,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        chunk = chunks[idx]

        retrieved_chunks.append({
            "chunk_id": int(idx),
            "text": chunk["text"],
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "score": float(score)
        })

    return retrieved_chunks