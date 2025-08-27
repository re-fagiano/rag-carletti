"""Ricostruisce l'indice FAISS a partire dai documenti in ``docs/``.

Per impostazione predefinita vengono indicizzati solo i file ``.txt``. Per
includere anche i PDF utilizzare l'opzione ``--include-pdf``: il testo verrà
estratto e suddiviso in sezioni rilevanti prima dell'embedding.
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import argparse
import os


def load_txt_documents() -> list:
    loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)
    return loader.load()


def load_pdf_sections() -> list:
    from langchain_community.document_loaders import PDFMinerLoader

    loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PDFMinerLoader)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    return [doc for doc in chunks if len(doc.page_content.strip()) >= 100]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ricostruisce l'indice FAISS.")
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Includi anche i PDF (estratti e suddivisi in sezioni)",
    )
    args = parser.parse_args()

    load_dotenv()

    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise Exception(
                "Devi impostare la variabile d'ambiente DEEPSEEK_API_KEY per usare OpenAIEmbeddings."
            )
        api_base = "https://api.deepseek.com"
        model = os.getenv("DEEPSEEK_EMBEDDINGS_MODEL", "deepseek-embedding")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception(
                "Devi impostare la variabile d'ambiente OPENAI_API_KEY per usare OpenAIEmbeddings."
            )
        api_base = os.getenv("OPENAI_API_BASE")
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")

    docs = load_txt_documents()
    if args.include_pdf:
        docs.extend(load_pdf_sections())

    embeddings = OpenAIEmbeddings(
        model=model, openai_api_key=api_key, openai_api_base=api_base
    )
    faiss_db = FAISS.from_documents(docs, embeddings)
    faiss_db.save_local("vectordb/")
    print("✅ Indice FAISS (dim=1536) ricostruito correttamente.")


if __name__ == "__main__":
    main()

