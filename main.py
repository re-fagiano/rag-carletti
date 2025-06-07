import logging
import traceback
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import RetrievalQA

# 1) Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# 2) Monta i tuoi file statici (la UI del bot)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# 3) Caricamento FAISS al boot (assicurati di aver committato vectordb/)
try:
    hf_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("vectordb/", hf_embedding, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    rag = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        retriever=retriever
    )
    logger.info("🔌 FAISS Retriever caricato correttamente.")
except Exception as e:
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
        # restituisco al client l’eccezione per debugging
        raise HTTPException(500, detail=f"Internal error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
