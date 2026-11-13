
from pathlib import Path
import json
import faiss
import urllib.request
import urllib.error

from sentence_transformers import SentenceTransformer

from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker
from src.answerability_checker import check_answerability


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

DATASETS = {
    "northstar": {
        "index": BASE_DIR / "vectorstore" / "enterprise_faiss.index",
        "chunks": BASE_DIR / "vectorstore" / "chunks_metadata.json"
    },

    "apex": {
        "index": (
            BASE_DIR / "data" / "benchmark_v2" /
            "processed" / "apex_faiss.index"
        ),
        "chunks": (
            BASE_DIR / "data" / "benchmark_v2" /
            "processed" / "apex_chunks_metadata.json"
        )
    }
}


# ============================================================
# 2. LOAD RAG COMPONENTS
# ============================================================

def load_rag_components(dataset="northstar"):

    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Choose from {list(DATASETS.keys())}"
        )

    paths = DATASETS[dataset]

    print(f"Loading dataset: {dataset}", flush=True)

    if not paths["index"].exists():
        raise FileNotFoundError(paths["index"])

    if not paths["chunks"].exists():
        raise FileNotFoundError(paths["chunks"])

    print("Loading embedding model...", flush=True)

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    index = faiss.read_index(str(paths["index"]))

    with open(
        paths["chunks"],
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError(
            "Vector index and chunk metadata count do not match."
        )

    return embedding_model, index, chunks


# ============================================================
# 3. GET CHUNK METADATA
# ============================================================

def get_chunk_metadata(chunk):

    metadata = chunk.get("metadata", {})

    return {
        "source": metadata.get(
            "source",
            chunk.get("source", "Unknown")
        ),
        "page": metadata.get(
            "page",
            chunk.get("page", "Unknown")
        )
    }


# ============================================================
# 4. RECONSTRUCT COMPLETE PAGE FROM CHUNKS
# ============================================================

def merge_overlapping_chunks(chunk_texts):

    if not chunk_texts:
        return ""

    merged_text = chunk_texts[0].strip()

    for next_text in chunk_texts[1:]:

        next_text = next_text.strip()

        if not next_text:
            continue

        if not merged_text:
            merged_text = next_text
            continue

        # Avoid adding text already contained in the page.
        if next_text in merged_text:
            continue

        # Find the largest overlap between the end of the
        # existing text and the beginning of the next chunk.
        max_overlap = min(
            len(merged_text),
            len(next_text),
            500
        )

        overlap = 0

        for size in range(max_overlap, 0, -1):

            if (
                merged_text[-size:]
                == next_text[:size]
            ):
                overlap = size
                break

        # Remove duplicated overlapping characters.
        remaining_text = next_text[overlap:]

        if remaining_text:

            if merged_text.endswith((" ", "\n")):
                merged_text += remaining_text
            elif remaining_text.startswith((" ", "\n")):
                merged_text += remaining_text
            else:
                merged_text += " " + remaining_text

    return merged_text.strip()


def reconstruct_page_context(
    original_chunks,
    validated_evidence
):

    reconstructed_pages = []

    seen_pages = set()

    for evidence in validated_evidence:

        source = evidence["source"]
        page = evidence["page"]

        page_key = (source, page)

        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)

        page_texts = []

        # Find every original chunk belonging to the
        # same source document and page.
        for chunk in original_chunks:

            metadata = get_chunk_metadata(chunk)

            if (
                metadata["source"] == source
                and metadata["page"] == page
            ):

                text = chunk.get("text", "").strip()

                if text:
                    page_texts.append(text)

        # Reconstruct the page using its original
        # chunk order and remove overlap duplicates.
        complete_page_text = merge_overlapping_chunks(
            page_texts
        )

        if complete_page_text:

            reconstructed_pages.append({
                "text": complete_page_text,
                "source": source,
                "page": page
            })

    return reconstructed_pages


# ============================================================
# 5. GENERATE ANSWER USING OLLAMA
# ============================================================

def generate_answer(question, retrieved_chunks):

    context = ""

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context += (
            f"\nDocument {i}:\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content: {chunk['text']}\n"
        )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the provided
document context.

IMPORTANT RULES:

1. Read the complete context before answering.
2. Extract the exact information relevant to the question.
3. Do not invent or assume information.
4. Do not ignore relevant information present in the context.
5. If the answer is explicitly available, answer it directly.
6. If the answer is genuinely unavailable, say:
   "I could not find this information in the provided documents."
7. Keep the answer clear and concise.
8. Do not generate source filenames or page numbers.
9. Do not invent citations.

CONTEXT:
{context}

USER QUESTION:
{question}

Provide the answer based strictly on the context.

ANSWER:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=300
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        answer = result["response"].strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty answer."
            )

        # Generate citations deterministically using
        # the actual source metadata.
        source_citations = []

        seen = set()

        for chunk in retrieved_chunks:

            citation = (
                chunk["source"],
                chunk["page"]
            )

            if citation not in seen:

                source_citations.append(citation)

                seen.add(citation)

        answer += "\n\nSources:"

        for source, page in source_citations:

            answer += (
                f"\n- {source}, Page {page}"
            )

        return answer

    except urllib.error.URLError as error:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        ) from error


# ============================================================
# 6. MAIN RAG PIPELINE
# ============================================================

def run_rag(
    question,
    dataset="northstar"
):

    print("\nLoading RAG components...", flush=True)

    embedding_model, index, chunks = load_rag_components(
        dataset
    )

    print(
        "\nInitializing hybrid retriever...",
        flush=True
    )

    hybrid_retriever = HybridRetriever(
        embedding_model,
        index,
        chunks
    )

    print(
        "\nInitializing Cross-Encoder reranker...",
        flush=True
    )

    reranker = CrossEncoderReranker()

    print(
        "\nPerforming hybrid retrieval...",
        flush=True
    )

    candidates = hybrid_retriever.search(
        question,
        top_k=10
    )

    print(
        "\nReranking retrieved documents...",
        flush=True
    )

    reranked_chunks = reranker.rerank(
        question,
        candidates,
        top_k=5
    )

    # --------------------------------------------------------
    # Convert retrieved chunks into a consistent format.
    # --------------------------------------------------------

    relevant_chunks = []

    for chunk in reranked_chunks:

        metadata = get_chunk_metadata(chunk)

        relevant_chunks.append({
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk["text"],
            "source": metadata["source"],
            "page": metadata["page"],
            "metadata": metadata,
            "rrf_score": chunk.get("rrf_score", 0.0),
            "rerank_score": chunk.get(
                "rerank_score",
                0.0
            )
        })

    # --------------------------------------------------------
    # Handle empty retrieval.
    # --------------------------------------------------------

    if not relevant_chunks:

        answer = (
            "I could not find this information "
            "in the provided documents."
        )

        verification = {
            "supported": False,
            "reason": "No relevant documents were retrieved.",
            "evidence": []
        }

    else:

        print(
            "\nChecking whether retrieved evidence supports the question...",
            flush=True
        )

        verification = check_answerability(
            question,
            relevant_chunks
        )

        print(
            "\nEvidence supported:",
            verification["supported"]
        )

        print(
            "Verification reason:",
            verification["reason"]
        )

        # ----------------------------------------------------
        # Generate answer only when evidence is supported.
        # ----------------------------------------------------

        if verification["supported"]:

            validated_evidence = verification.get(
                "evidence",
                []
            )

            if not validated_evidence:

                answer = (
                    "I could not find this information "
                    "in the provided documents."
                )

                verification["supported"] = False

                verification["reason"] = (
                    "No validated evidence was available "
                    "for answer generation."
                )

            else:

                print(
                    "\nReconstructing source page context...",
                    flush=True
                )

                evidence_chunks = reconstruct_page_context(
                    chunks,
                    validated_evidence
                )

                # Safe fallback if complete page context
                # cannot be reconstructed.
                if not evidence_chunks:

                    print(
                        "Page reconstruction unavailable. "
                        "Using validated evidence instead.",
                        flush=True
                    )

                    evidence_chunks = []

                    for item in validated_evidence:

                        evidence_chunks.append({
                            "text": item["statement"],
                            "source": item["source"],
                            "page": item["page"]
                        })

                print(
                    "\nGeneration context prepared:",
                    flush=True
                )

                for item in evidence_chunks:

                    print(
                        f"- {item['source']} | "
                        f"Page {item['page']} | "
                        f"{len(item['text'])} characters"
                    )

                print(
                    "\nGenerating answer using source page context...",
                    flush=True
                )

                answer = generate_answer(
                    question,
                    evidence_chunks
                )

        else:

            answer = (
                "I could not find this information "
                "in the provided documents."
            )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    validated_evidence = verification.get(
        "evidence",
        []
    )

    print("\n" + "=" * 60)

    print("ADVANCED ENTERPRISE RAG AI ASSISTANT")

    print("=" * 60)

    print("\nDATASET:")
    print(dataset)

    print("\nQUESTION:")
    print(question)

    print("\nAI GENERATED ANSWER:")
    print(answer)

    # ==========================================
    # VALIDATED EVIDENCE SOURCES
    # ==========================================

    print("\nVALIDATED EVIDENCE SOURCES:")

    if verification["supported"] and validated_evidence:

        seen_sources = set()

        for item in validated_evidence:

            source_key = (
                item["source"],
                item["page"]
            )

            if source_key in seen_sources:
                continue

            seen_sources.add(source_key)

            print(
                f"- {item['source']} | "
                f"Page {item['page']}"
            )

    else:

        print("No verified evidence sources.")

    # ==========================================
    # ALL RETRIEVED SOURCES
    # ==========================================

    print("\nALL RETRIEVED SOURCES:")

    for chunk in relevant_chunks:

        print(
            f"- {chunk['source']} | "
            f"Page {chunk['page']} | "
            f"RRF Score: {chunk['rrf_score']:.4f} | "
            f"Rerank Score: {chunk['rerank_score']:.4f}"
        )

    # ==========================================
    # EVIDENCE VERIFICATION
    # ==========================================

    print("\nEVIDENCE VERIFICATION:")
    print(verification["reason"])

    print(
        "\nAdvanced RAG pipeline completed successfully!"
    )

    return {
        "answer": answer,
        "supported": verification["supported"],
        "reason": verification["reason"],
        "sources": (
            validated_evidence
            if verification["supported"]
            else []
        ),
        "retrieved_sources": relevant_chunks
    }