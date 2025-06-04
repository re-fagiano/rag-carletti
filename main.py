from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# 1) Monta la cartella "static" su "/static"
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2) Definisci la rotta GET per la root "/"
#    Serve il file static/index.html quando visiti http://127.0.0.1:8000/
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# 3) Prepara la RAG (viene caricato al momento dell'avvio del server)
hf_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectordb/", hf_embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
rag = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4"),
    retriever=retriever
)

# 4) Endpoint POST "/ask" che restituisce il JSON { rispsta: "..." }
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("query", "")
    print(f"[LOG] Query ricevuta: '{query}'")  # log per debug
    try:
        risposta = rag.run(query)
        print(f"[LOG] Risposta generata: '{risposta}'")
    except Exception as e:
        print(f"[ERROR] Errore in rag.run(): {e}")
        risposta = "❌ Errore interno. Controlla i log del server."
    return {"risposta": risposta}
