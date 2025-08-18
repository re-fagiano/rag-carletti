from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

# Leggi chiave
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Prepara documenti
docs = []
for file in Path("docs/").rglob("*.txt"):
    text = file.read_text(encoding="utf8")
    docs.append(Document(page_content=text, metadata={"source": str(file)}))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150
)
split_docs = text_splitter.split_documents(docs)

# Embeddings e FAISS
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
faiss_db = FAISS.from_documents(split_docs, embeddings)
faiss_db.save_local("vectordb/")
print("✅ Indice FAISS (dim=1536) ricostruito correttamente.")
