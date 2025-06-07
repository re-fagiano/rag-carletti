import logging
import traceback
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Usiamo solo OpenAIEmbeddings per evitare OOM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Inizializza FastAPI
app = FastAPI()

# Servi la pagina HTML principale
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Monta i file statici su /static (JS, CSS, assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Leggi chiave API da variabili d'ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Devi impostare la variabile d'ambiente OPENAI_API_KEY")

# Inizializza la pipeline RAG
try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local("vectordb/", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    logger.info("🔌 FAISS Retriever caricato correttamente con OpenAIEmbeddings.")
except Exception:
    logger.exception("❌ Errore durante il caricamento di FAISS o OpenAI Embeddings:")
    raise

# Endpoint salute
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint chat
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
        raise
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"❌ Errore interno durante /ask:\n{tb}")
        from fastapi.responses import JSONResponse
        
        # Temporaneo: ritorna stacktrace nella risposta JSON
        return JSONResponse(status_code=500, content={"error": tb})
