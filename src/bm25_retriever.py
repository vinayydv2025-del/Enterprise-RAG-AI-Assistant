
import re
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):

        self.chunks = chunks

        self.tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    @staticmethod
    def tokenize(text):

        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def search(self, query, top_k=5):

        tokenized_query = self.tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            results.append({
                "chunk_id": int(index),
                "text": self.chunks[index]["text"],
                "metadata": self.chunks[index]["metadata"],
                "score": float(scores[index])
            })

        return results
