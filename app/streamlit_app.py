
import os
import sys
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="Advanced Enterprise RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. CLOUD SECRETS AND GROQ CONFIGURATION
# ============================================================

try:
    cloud_api_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    cloud_api_key = ""

if cloud_api_key:
    os.environ["GROQ_API_KEY"] = str(cloud_api_key)

# This Streamlit application always uses Groq.
os.environ["LLM_PROVIDER"] = "groq"

for secret_name in (
    "GROQ_MODEL",
    "GROQ_BASE_URL",
    "GROQ_TIMEOUT"
):
    try:
        secret_value = st.secrets.get(secret_name)
    except Exception:
        secret_value = None

    if secret_value:
        os.environ[secret_name] = str(secret_value)


# Import after configuring environment variables.
from src.rag_pipeline import (
    DATASETS,
    GROQ_API_KEY,
    load_rag_components,
    run_rag_with_components,
)

from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker


# ============================================================
# 3. CACHE MODELS AND RETRIEVAL COMPONENTS
# ============================================================

@st.cache_resource(max_entries=1, show_spinner=False)
def load_dataset_resources(dataset_name):

    embedding_model, index, chunks = load_rag_components(
        dataset_name
    )

    hybrid_retriever = HybridRetriever(
        embedding_model,
        index,
        chunks
    )

    return (
        embedding_model,
        index,
        chunks,
        hybrid_retriever
    )


@st.cache_resource(show_spinner=False)
def load_reranker():

    return CrossEncoderReranker()


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.title("🤖 Advanced Enterprise RAG AI Assistant")

st.write(
    "Ask questions about company policies and receive "
    "document-grounded answers with evidence verification."
)

st.caption(
    "Powered by FAISS, BM25, Hybrid Retrieval, "
    "Cross-Encoder Reranking and Groq."
)

st.divider()


# ============================================================
# 5. USER INPUT
# ============================================================

st.subheader("Ask your question")

dataset = st.selectbox(
    "Select knowledge base:",
    options=list(DATASETS.keys()),
    format_func=lambda value: value.title()
)

question = st.text_input(
    "Enter your question:",
    placeholder="Example: How many annual leave days do employees receive?"
)


# ============================================================
# 6. EXECUTE RAG PIPELINE
# ============================================================

if st.button("Ask AI", type="primary", use_container_width=True):

    if not question.strip():

        st.warning("Please enter a question.")

    elif len(question.strip()) < 3:

        st.warning("Please enter a more detailed question.")

    elif not GROQ_API_KEY:

        st.error(
            "Groq API key is missing. "
            "Configure GROQ_API_KEY in your environment or "
            "Streamlit Secrets."
        )

    else:

        try:

            with st.spinner(
                "Loading retrieval models and knowledge base..."
            ):

                (
                    embedding_model,
                    index,
                    chunks,
                    hybrid_retriever
                ) = load_dataset_resources(dataset)

                reranker = load_reranker()

            with st.spinner(
                "Retrieving documents, reranking and verifying evidence..."
            ):

                result = run_rag_with_components(
                    question=question.strip(),
                    dataset=dataset,
                    embedding_model=embedding_model,
                    index=index,
                    chunks=chunks,
                    hybrid_retriever=hybrid_retriever,
                    reranker=reranker
                )

            # -----------------------------------------------
            # ANSWER
            # -----------------------------------------------

            st.divider()

            st.subheader("AI Generated Answer")

            st.write(result["answer"])

            # -----------------------------------------------
            # EVIDENCE VERIFICATION
            # -----------------------------------------------

            st.subheader("Evidence Verification")

            if result["supported"]:

                st.success("Evidence supported")

            else:

                st.warning("Evidence not found")

            st.write(
                "**Verification reason:**",
                result["reason"]
            )

            # -----------------------------------------------
            # VALIDATED SOURCES
            # -----------------------------------------------

            st.subheader("Validated Evidence Sources")

            sources = result.get("sources", [])

            if sources:

                for source in sources:

                    label = (
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )

                    with st.expander(label):

                        if source.get("statement"):

                            st.write(
                                source["statement"]
                            )

            else:

                st.info("No validated sources found.")

            # -----------------------------------------------
            # RETRIEVAL DIAGNOSTICS
            # -----------------------------------------------

            with st.expander("Retrieval Diagnostics"):

                retrieved = result.get(
                    "retrieved_sources",
                    []
                )

                if retrieved:

                    for item in retrieved:

                        st.write(
                            f"**{item['source']}** | "
                            f"Page {item['page']}"
                        )

                        st.caption(
                            f"RRF Score: {item.get('rrf_score', 0):.4f} | "
                            f"Cross-Encoder Score: "
                            f"{item.get('rerank_score', 0):.4f}"
                        )

                else:

                    st.info("No documents retrieved.")

        except Exception as error:

            st.error(
                f"RAG processing failed: {error}"
            )


# ============================================================
# 7. FOOTER
# ============================================================

st.divider()

st.caption(
    "Python | FAISS | BM25 | Sentence Transformers | "
    "Cross-Encoder | FastAPI-compatible RAG pipeline | Groq"
)



