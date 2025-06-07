import logging
import traceback
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles

# Usiamo solo OpenAIEmbeddings per evitare OOM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Caricamento FastAPI e file statici
app = FastAPI()t("/", StaticFiles(directory="static", html=True), name="static")

# Leggi chiave API da variabili d'ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Devi impostare la variabile d'ambiente OPENAI_API_KEY")

try:
    # Inizializza embeddings OpenAI (remote)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Carica il database FAISS locale, usando gli embeddings OpenAI
    db = FAISS.load_local(
        "vectordb/", embeddings, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever()

    # Inizializza il modello LLM ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

    # Costruisci la catena RetrievalQA
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    logger.info("🔌 FAISS Retriever caricato correttamente con OpenAIEmbeddings.")
except Exception:
    logger.exception("❌ Errore durante il caricamento di FAISS o OpenAI Embeddings:")
    raise

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(request: Request):
    try:
        payload = await request.json()
        query = payload.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=422, detail="Inserisci il campo 'query' nel JSON")
        logger.info(f"▶️ Ricevuta query: {query!r}")
        risposta = rag.run(query)
        logger.info(f"✅ Risposta: {risposta!r}")
        return {"risposta": risposta}
    except HTTPException:
        # rilancia HTTPException per status 422
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Errore interno durante /ask:\n{tb}")
        raise HTTPException(status_code=500, detail="Internal error, controlla i log")

app.mounor, controlla i log")
