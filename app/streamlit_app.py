
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="Advanced Enterprise RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced Enterprise RAG AI Assistant")

st.write(
    "Ask questions about company policies and receive "
    "document-grounded answers with evidence verification."
)

st.divider()

question = st.text_input(
    "Enter your question:",
    placeholder="Example: How many annual leave days do employees receive?"
)

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
                    json={"question": question},
                    timeout=300
                )

                response.raise_for_status()

                result = response.json()

                st.subheader("AI Generated Answer")

                st.write(result["answer"])

                st.subheader("Evidence Verification")

                if result["evidence_supported"]:

                    st.success("Evidence supported")

                else:

                    st.warning("Evidence not found")

                st.write(
                    "**Verification reason:**",
                    result["verification_reason"]
                )

                st.subheader("Retrieved Sources")

                if result["sources"]:

                    for source in result["sources"]:

                        with st.expander(
                            f"📄 {source['source']} — Page {source['page']}"
                        ):

                            st.write(
                                f"**RRF Score:** {source['rrf_score']}"
                            )

                            st.write(
                                f"**Cross-Encoder Score:** "
                                f"{source['rerank_score']}"
                            )

                else:

                    st.info("No relevant sources found.")

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to the API: {error}"
                )

            except (KeyError, ValueError) as error:

                st.error(
                    f"Unexpected API response: {error}"
                )

st.divider()

st.caption(
    "Powered by Python | FAISS | BM25 | "
    "Cross-Encoder | Sentence Transformers | Ollama"
)

