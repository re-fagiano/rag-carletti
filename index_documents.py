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
import re


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
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "deepseek"),
        choices=["openai", "deepseek"],
        help="Provider LLM da utilizzare (default: deepseek)",
    )
    args = parser.parse_args()

    load_dotenv()

    provider = args.provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            api_key = re.sub(r"\s+", "", api_key or "")
        if not api_key:
            raise Exception(
                "Devi impostare la variabile d'ambiente OPENAI_API_KEY oppure usare --provider deepseek"
            )
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    else:  # deepseek
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            api_key = re.sub(r"\s+", "", api_key or "")
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

    documents = load_txt_documents()
    if args.include_pdf:
        documents.extend(load_pdf_sections())

    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vectordb/")
    print(f"✅ Indicizzazione completata utilizzando il provider {provider}.")


if __name__ == "__main__":
    main()

