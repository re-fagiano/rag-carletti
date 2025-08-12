from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

load_dotenv()

# Leggi chiave
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Modello di embedding configurabile tramite variabile d'ambiente
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# Prepara documenti
docs = []
for file in Path("docs/").rglob("*.txt"):
    text = file.read_text(encoding="utf8")
    docs.append(Document(page_content=text, metadata={"source": str(file)}))

# Embeddings e FAISS
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL, openai_api_key=os.getenv("OPENAI_API_KEY")
)
faiss_db = FAISS.from_documents(docs, embeddings)
faiss_db.save_local("vectordb/")
print(f"✅ Indice FAISS ricostruito con il modello {EMBED_MODEL}.")
