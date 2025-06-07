import logging, traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles

# usa gli embedding e FAISS dal core (o da community, va bene)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# il modello chat
from langchain_community.chat_models import ChatOpenAI

# **qui** import corretto
from langchain.chains import RetrievalQA

# --- resto identico ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")

try:
    hf_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("vectordb/", hf_embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    rag = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=retriever
    )
    logger.info("🔌 FAISS Retriever caricato correttamente.")
except Exception:
    logger.exception("❌ Errore durante il caricamento di FAISS:")
    raise

@app.post("/ask")
async def ask_question(request: Request):
    try:
        payload = await request.json()
        query = payload.get("query", "").strip()
        logger.info(f"▶️ Ricevuta query: {query!r}")
        risposta = rag.run(query)
        logger.info(f"✅ Risposta: {risposta!r}")
        return {"risposta": risposta}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Errore interno durante /ask:\n{tb}")
        raise HTTPException(500, detail=f"Internal error: {e}")
