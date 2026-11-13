
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import retrieve_documents


class HybridRetriever:

    def __init__(self, embedding_model, index, chunks):

        self.embedding_model = embedding_model
        self.index = index
        self.chunks = chunks

        self.bm25_retriever = BM25Retriever(chunks)

    def search(self, query, top_k=5, rrf_k=60):

        # Dense retrieval using FAISS
        dense_results = retrieve_documents(
            query,
            self.embedding_model,
            self.index,
            self.chunks,
            top_k=top_k
        )

        # Sparse retrieval using BM25
        sparse_results = self.bm25_retriever.search(
            query,
            top_k=top_k
        )

        # Reciprocal Rank Fusion
        rrf_scores = {}

        for rank, result in enumerate(dense_results, start=1):

            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

        for rank, result in enumerate(sparse_results, start=1):

            chunk_id = result["chunk_id"]

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (rrf_k + rank)
            )

        # Sort chunks by combined RRF score
        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )[:top_k]

        final_results = []

        for chunk_id in ranked_ids:

            chunk = self.chunks[chunk_id]

            final_results.append({
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "rrf_score": rrf_scores[chunk_id]
            })

        return final_results

