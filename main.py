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

load_dotenv()  # in produzione Render caricherà OPENAI_API_KEY da env vars

app = FastAPI()

# (Opzionale) Abilita CORS se frontend e backend stanno su domini diversi:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o ["https://tuo-dominio-wordpress.it"]
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Monta static/ su /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# GET / → serve index.html
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# 1) Usa OpenAIEmbeddings (richiede che OPENAI_API_KEY sia impostato in env)
openai_embedding = OpenAIEmbeddings()
db = FAISS.load_local("vectordb/", openai_embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# 2) Usa il LLM di OpenAI (ChatOpenAI) via API
rag = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),  # o "gpt-4" se hai accesso
    retriever=retriever
)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = data.get("query", "")
    try:
        risposta = rag.run(query)
    except Exception as e:
        risposta = "❌ Errore interno: " + str(e)
    return {"risposta": risposta}

