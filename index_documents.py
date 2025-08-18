"""Script per indicizzare i documenti nella cartella ``docs/``.

Per impostazione predefinita vengono elaborati soltanto i file ``.txt``.
È possibile includere anche i PDF passando l'opzione ``--include-pdf``.
In questo caso il testo viene estratto e suddiviso in sezioni rilevanti prima
di generare le embedding, così da evitare di indicizzare parti superflue del
documento.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import os


def load_txt_documents() -> list:
    """Carica tutti i file ``.txt`` presenti nella cartella ``docs/``."""
    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def load_pdf_sections() -> list:
    """Estrae il testo dai PDF e lo suddivide in sezioni rilevanti.

    Le sezioni con meno di 100 caratteri vengono scartate per ridurre il rumore
    nell'indice finale.
    """

    from langchain_community.document_loaders import PDFMinerLoader

    loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PDFMinerLoader)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    return [doc for doc in chunks if len(doc.page_content.strip()) >= 100]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indicizza i documenti presenti in docs/."
    )
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Includi anche i PDF (estratti e suddivisi in sezioni)",
    )
    args = parser.parse_args()

    load_dotenv()

    # Assicurati che la variabile d'ambiente OPENAI_API_KEY sia disponibile
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise Exception(
            "Devi impostare la variabile d'ambiente OPENAI_API_KEY per usare OpenAIEmbeddings."
        )

    documents = load_txt_documents()
    if args.include_pdf:
        documents.extend(load_pdf_sections())

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vectordb/")
    print("✅ Indicizzazione completata utilizzando OpenAIEmbeddings.")


if __name__ == "__main__":
    main()

