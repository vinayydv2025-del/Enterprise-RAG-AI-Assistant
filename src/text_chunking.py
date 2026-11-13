
from src.document_ingestion import load_pdf_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(data_dir=None):

    if data_dir is None:
        documents = load_pdf_documents()
    else:
        documents = load_pdf_documents(data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(document["text"])

        for chunk_number, text in enumerate(text_chunks, start=1):

            chunks.append({
                "text": text,
                "metadata": {
                    **document["metadata"],
                    "chunk_number": chunk_number
                }
            })

    return chunks


if __name__ == "__main__":

    chunks = split_documents()

    print("\n" + "=" * 50)
    print("TEXT CHUNKING SUMMARY")
    print("=" * 50)

    print("Total chunks created:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(f"\nCHUNK {index}")
        print("Source:", chunk["metadata"]["source"])
        print("Page:", chunk["metadata"]["page"])
        print("Chunk number:", chunk["metadata"]["chunk_number"])
        print("Text:", chunk["text"])

    print("\nText chunking completed successfully!")
