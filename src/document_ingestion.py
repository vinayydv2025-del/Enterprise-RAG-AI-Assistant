
from pathlib import Path
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = BASE_DIR / "data" / "raw"


def load_pdf_documents(data_dir=DEFAULT_DATA_DIR):
    documents = []

    data_dir = Path(data_dir)

    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}"
        )

    for pdf_path in pdf_files:

        reader = PdfReader(str(pdf_path))

        print(f"\nReading: {pdf_path.name}")

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():

                documents.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number
                    }
                })

                print(
                    f"Page {page_number}: "
                    f"{len(text)} characters extracted"
                )

    return documents


if __name__ == "__main__":

    docs = load_pdf_documents()

    print("\n" + "=" * 50)
    print("DOCUMENT INGESTION SUMMARY")
    print("=" * 50)

    print(f"Total pages extracted: {len(docs)}")

    for index, doc in enumerate(docs, start=1):
        print(f"\nDocument {index}")
        print("Source:", doc["metadata"]["source"])
        print("Page:", doc["metadata"]["page"])
        print("Text preview:", doc["text"][:200])

    print("\nPDF ingestion completed successfully!")
