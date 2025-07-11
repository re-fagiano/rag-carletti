import logging
import traceback
import os
import re
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
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
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("Devi impostare la variabile d'ambiente OPENAI_API_KEY")

VECTORDB_PATH = "vectordb/"
if not os.path.isdir(VECTORDB_PATH):
    raise Exception(f"Directory '{VECTORDB_PATH}' non trovata. Ricrea o committa l'indice FAISS.")

try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

    system_instruction = (
        "Sei un tecnico professionale per la riparazione di elettrodomestici. Rispondi sempre in modo chiaro, tecnico, e senza ironia. "
        "Non aggiungere battute, frasi umoristiche o riferimenti surreali. Concentrati solo sulla risoluzione del problema. "
        "Se rilevi termini tecnici, formattali con i tooltip. Se opportuno, includi un'immagine rilevante tramite Bing."
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instruction),
        HumanMessagePromptTemplate.from_template("Contesto:\n{context}\n\nDomanda: {question}"),
    ])

    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
    )
    logger.info("✅ Pipeline RAG inizializzata correttamente.")
except Exception:
    logger.exception("❌ Errore durante l'inizializzazione della pipeline RAG:")
    raise

TOOLTIPS = {
    "filtro": "Componente da pulire regolarmente per evitare intasamenti e cattivi odori.",
    "filtri": "Componenti da pulire regolarmente per evitare intasamenti e cattivi odori.",
    "scarico": "Il sistema che espelle l'acqua dalla lavatrice o lavastoviglie.",
    "pompa": "Dispositivo che serve a espellere l'acqua dall’elettrodomestico.",
    "motore": "Cuore del funzionamento meccanico, può essere inverter o tradizionale.",
    "pressostato": "Sensore di pressione che regola il livello dell’acqua.",
    "elettrovalvola": "Componente che apre o chiude il passaggio dell'acqua.",
    "resistenza": "Serve a riscaldare l’acqua nei cicli di lavaggio.",
    "guarnizione": "Elemento in gomma per prevenire perdite di acqua.",
    "scheda elettronica": "Il cervello dell’elettrodomestico: gestisce tutte le funzioni.",
    "errore": "Indicazione di guasto tramite codice alfanumerico.",
    "codice errore": "Sigla (es: E10, F06) che indica un malfunzionamento specifico.",
    "codici errore": "Serie di sigle usate per indicare malfunzionamenti tecnici.",
    "tastiera": "Interfaccia utente: pulsanti e manopole.",
    "programma": "Ciclo di lavaggio o asciugatura selezionato dall’utente.",
    "inverter": "Tipo di motore elettronico a basso consumo."
}


def applica_tooltip(testo: str) -> str:
    for chiave, spiegazione in TOOLTIPS.items():
        pattern = r'(?<![\\w>])(' + re.escape(chiave) + r')(?![\\w<])'
        replacement = (
            r'<span class="tooltip">\1 <span class="info-icon">ⓘ</span>'
            r'<span class="tooltiptext">' + spiegazione + r'</span>'
            r'</span>'
        )
        testo = re.sub(pattern, replacement, testo, flags=re.IGNORECASE)
    return testo

def cerca_immagine_bing(query):
    if not BING_SEARCH_API_KEY:
        return ""
    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
    params = {"q": query, "count": 1, "imageType": "Photo"}
    response = requests.get("https://api.bing.microsoft.com/v7.0/images/search", headers=headers, params=params)
    try:
        results = response.json()
        return results["value"][0]["contentUrl"] if results["value"] else ""
    except:
        return ""

@app.post("/ask")
async def ask_question(request: Request):
    try:
        payload = await request.json()
        user_question = payload.get("query", "").strip()

        if not user_question:
            raise HTTPException(status_code=422, detail="Inserisci il campo 'query' nel JSON")

        logger.info(f"▶️ Ricevuta query: {user_question!r}")

        try:
            answer = rag.run(user_question)
        except AssertionError as ae:
            msg = "Indice FAISS non compatibile. Ricostruisci 'vectordb/' con lo stesso modello di embedding."
            return JSONResponse(status_code=500, content={"error": msg})

        image_url = cerca_immagine_bing(user_question)
        html_answer = answer.replace("\n", "<br>")
        html_answer = applica_tooltip(html_answer)

        if image_url:
            html_answer += f"<br><br><img src='{image_url}' alt='immagine correlata' style='max-width:100%; border-radius:8px;'>"

        return {"risposta": html_answer}

    except HTTPException:
        raise
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"❌ Errore interno durante /ask:\n{tb}")
        return JSONResponse(status_code=500, content={"error": tb})

@app.get("/health")
async def health():
    return {"status": "ok"}
