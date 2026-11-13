
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):

        print("Loading Cross-Encoder model...")

        self.model = CrossEncoder(
            model_name,
            device="cpu"
        )

        print("Cross-Encoder loaded successfully.")

    def rerank(self, query, retrieved_chunks, top_k=3):

        if not retrieved_chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for chunk, score in zip(retrieved_chunks, scores):

            result = chunk.copy()
            result["rerank_score"] = float(score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]
