import logging
import traceback
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS  # nuovo import consigliato
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
# ────────────────────────────────────────────────────────────────────────────────
# FastAPI bootstrap
# ────────────────────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
async def root():
    """Serve una semplice pagina HTML di test."""
    return FileResponse("static/index.html")

# File statici (JS/CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ────────────────────────────────────────────────────────────────────────────────
# Variabili d'ambiente obbligatorie
# ────────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Devi impostare la variabile d'ambiente OPENAI_API_KEY")

# ────────────────────────────────────────────────────────────────────────────────
# Carica indice FAISS
# ────────────────────────────────────────────────────────────────────────────────
VECTORDB_PATH = "vectordb/"
if not os.path.isdir(VECTORDB_PATH):
    raise Exception(
        f"Directory '{VECTORDB_PATH}' non trovata. Ricrea o committa l'indice FAISS."
    )

try:
    # Embeddings + retriever
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

    # ────────────────────────────────────────────────────────────────────────
    # Prompt di sistema
    # ────────────────────────────────────────────────────────────────────────
system_instruction = 
    "Sei un operatore di callcenter in una ditta di riparazioni di elettrodomestici. "
    "Il tuo obiettivo è aiutare i clienti con la riparazione dei loro elettrodomestici. "
    "Dimostra sempre grande competenza e professionalità, con un tono ironico e canzonatorio. "
    "Hai molte informazioni riguardanti le lavatrici e ti piace aiutare anche i principianti."

    # Prompt template (context + question)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_instruction),
            HumanMessagePromptTemplate.from_template(
                "Contesto:\n{context}\n\nDomanda: {question}"
            ),
        ]
    )

    # Retrieval‑Augmented chain (stuff)x1
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",  # dove inseriremo i docs
        },
    )
    logger.info("✅ Pipeline RAG inizializzata correttamente.")
except Exception:  # pragma: no cover
    logger.exception("❌ Errore durante l'inizializzazione della pipeline RAG:")
    raise

# ────────────────────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Semplice health‑check."""
    return {"status": "ok"}


@app.post("/ask")
async def ask_question(request: Request):
    """Endpoint principale di chat."""
    try:
        payload = await request.json()
        user_question = payload.get("query", "").strip()

        if not user_question:
            raise HTTPException(status_code=422, detail="Inserisci il campo 'query' nel JSON")

        logger.info(f"▶️ Ricevuta query: {user_question!r}")

        # Esegui la RAG
        try:
            answer = rag.run(user_question)
        except AssertionError as ae:
            msg = (
                "Indice FAISS non compatibile: dimensione embedding mismatch. "
                "Ricostruisci 'vectordb/' con lo stesso modello di embedding."
            )
            logger.error(f"❌ {msg}: {ae}")
            return JSONResponse(status_code=500, content={"error": msg})

        logger.info(f"✅ Risposta: {answer!r}")
        return {"risposta": answer}

    except HTTPException:
        raise
    except Exception:  # pragma: no cover
        tb = traceback.format_exc()
        logger.error(f"❌ Errore interno durante /ask:\n{tb}")
        return JSONResponse(status_code=500, content={"error": tb})
