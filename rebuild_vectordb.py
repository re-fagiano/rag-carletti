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
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "openai"),
        choices=["openai", "deepseek"],
        help="Provider LLM da utilizzare (default: valore di LLM_PROVIDER)",
    )
    args = parser.parse_args()

    load_dotenv()

    provider = args.provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception(
                "Devi impostare la variabile d'ambiente OPENAI_API_KEY oppure usare --provider deepseek"
            )
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    else:  # deepseek
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        if not api_key:
            raise Exception(
                "Devi impostare la variabile d'ambiente DEEPSEEK_API_KEY per usare --provider deepseek"
            )
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            base_url=base_url,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    docs = load_txt_documents()
    if args.include_pdf:
        docs.extend(load_pdf_sections())

    faiss_db = FAISS.from_documents(docs, embeddings)
    faiss_db.save_local("vectordb/")
    print(f"✅ Indice FAISS ricostruito con il provider {provider}.")


if __name__ == "__main__":
    main()

