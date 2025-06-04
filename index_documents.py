from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# Se hai PDF, assicurati di avere PDFMinerLoader installato:
# pip install langchain-community
# da langchain_community.document_loaders import PDFMinerLoader

load_dotenv()

# 1) Carica tutti i file .txt in documenti/ (ricorsivamente)
txt_loader = DirectoryLoader("documenti", glob="**/*.txt", loader_cls=TextLoader)

# 2) Se vuoi includere anche i PDF, decommenta le righe seguenti:
# pdf_loader = DirectoryLoader("documenti", glob="**/*.pdf", loader_cls=PDFMinerLoader)
# documents = txt_loader.load() + pdf_loader.load()

# Se ti bastano solo .txt, usa:
documents = txt_loader.load()

# 3) Scegli l’embedding locale (Sentence-Transformers)
hf_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4) Crea (o ricrea) il database FAISS
db = FAISS.from_documents(documents, hf_embedding)

# 5) Salva il vector store (sovrascrive la vecchia versione)
db.save_local("vectordb/")

print("✅ Indicizzazione (con tutti i documenti) completata.")
