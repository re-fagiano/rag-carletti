from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Carica la variabile OPENAI_API_KEY dal file .env (o dalle env vars di Render)
load_dotenv()

app = FastAPI()

# Se servirai il front-end da un dominio diverso, abilita CORS; altrimenti puoi ometterlo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in produzione sostituisci con il dominio del tuo WordPress
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

# Monta la cartella static/ sotto /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html sulla root
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# **Qui usiamo OpenAIEmbeddings invece di HuggingFaceEmbeddings**
openai_embedding = OpenAIEmbeddings()  
db = FAISS.load_local("vectordb/", openai_embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()
# Usa un modello GPT via API (puoi mettere "gpt-3.5-turbo" se non hai GPT-4)
rag = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    retriever=retriever
)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("query", "")
    try:
        risposta = rag.run(query)
    except Exception as e:
        risposta = f"❌ Errore interno: {e}"
    return {"risposta": risposta}
