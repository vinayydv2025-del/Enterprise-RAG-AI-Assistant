
import os
import streamlit as st
import requests


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

API_URL = f"{API_BASE_URL}/ask"

REQUEST_TIMEOUT = 300


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Advanced Enterprise RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Advanced Enterprise RAG AI Assistant")

st.write(
    "Ask questions about company policies and receive "
    "document-grounded answers with evidence verification."
)

st.caption("Powered by Hybrid Retrieval, BM25, FAISS and Cross-Encoder Reranking")

st.divider()


# ============================================================
# API CONNECTION
# ============================================================

with st.expander("API Configuration"):

    st.write(f"**API Endpoint:** `{API_URL}`")


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Enter your question:",
    placeholder="Example: How many annual leave days do employees receive?"
)


# ============================================================
# ASK AI
# ============================================================

if st.button("Ask AI", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Retrieving documents, reranking and verifying evidence..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={"question": question.strip()},
                    timeout=REQUEST_TIMEOUT
                )

                response.raise_for_status()

                result = response.json()

                # Validate required response fields
                required_fields = [
                    "answer",
                    "evidence_supported",
                    "verification_reason",
                    "sources"
                ]

                missing_fields = [
                    field for field in required_fields
                    if field not in result
                ]

                if missing_fields:

                    st.error(
                        f"API response is missing fields: {missing_fields}"
                    )

                else:

                    # ----------------------------------------
                    # ANSWER
                    # ----------------------------------------

                    st.subheader("AI Generated Answer")

                    st.write(result["answer"])

                    # ----------------------------------------
                    # EVIDENCE VERIFICATION
                    # ----------------------------------------

                    st.subheader("Evidence Verification")

                    if result["evidence_supported"]:

                        st.success("Evidence supported")

                    else:

                        st.warning("Evidence not found")

                    st.write(
                        "**Verification reason:**",
                        result["verification_reason"]
                    )

                    # ----------------------------------------
                    # SOURCES
                    # ----------------------------------------

                    st.subheader("Retrieved Sources")

                    if result["sources"]:

                        for source in result["sources"]:

                            with st.expander(
                                f"📄 {source['source']} — Page {source['page']}"
                            ):

                                st.write(
                                    f"**RRF Score:** "
                                    f"{source.get('rrf_score', 'N/A')}"
                                )

                                st.write(
                                    f"**Cross-Encoder Score:** "
                                    f"{source.get('rerank_score', 'N/A')}"
                                )

                    else:

                        st.info("No relevant sources found.")

            except requests.exceptions.Timeout:

                st.error(
                    "The API request timed out. "
                    "The RAG pipeline may need more processing time."
                )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI backend. "
                    "Check that the API server is running and API_URL is correct."
                )

            except requests.exceptions.HTTPError as error:

                st.error(
                    f"API returned an HTTP error: {error}"
                )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Request failed: {error}"
                )

            except (ValueError, KeyError, TypeError) as error:

                st.error(
                    f"Unexpected API response: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Python | FAISS | BM25 | "
    "Cross-Encoder | Sentence Transformers | Ollama"
)


