from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings # Modifica qui
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os # Importa il modulo os

# Se hai PDF, assicurati di avere PDFMinerLoader installato:
# pip install langchain-community[pdf]
# da langchain_community.document_loaders import PDFMinerLoader

load_dotenv()

# Assicurati che la variabile d'ambiente OPENAI_API_KEY sia disponibile
# Questo è necessario per OpenAIEmbeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Devi impostare la variabile d'ambiente OPENAI_API_KEY per usare OpenAIEmbeddings.")

# 1) Carica tutti i file .txt in docs/ (ricorsivamente)
txt_loader = DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader)

# 2) Se vuoi includere anche i PDF, decommenta le righe seguenti e assicurati
#    di avere PDFMinerLoader installato:
# pdf_loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PDFMinerLoader)
# documents = txt_loader.load() + pdf_loader.load()

# Se ti bastano solo .txt, usa:
documents = txt_loader.load()

# 3) Scegli l’embedding di OpenAI
# Modifica qui: usa OpenAIEmbeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 4) Crea (o ricrea) il database FAISS
db = FAISS.from_documents(documents, embeddings)

# 5) Salva il vector store (sovrascrive la vecchia versione)
db.save_local("vectordb/")

print("✅ Indicizzazione (con tutti i documenti) completata utilizzando OpenAIEmbeddings.")
