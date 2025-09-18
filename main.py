import json
import logging
import os
import re
import asyncio
import traceback
from html import escape
import requests
import openai
from datetime import datetime
from types import MappingProxyType
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, conint
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


@app.get("/debug/ping-deepseek")
async def debug_ping_deepseek():
    """Ping di debug verso l'endpoint DeepSeek /v1/models."""
    ping_url = f"{DEEPSEEK_API_BASE}/models"
    if not DEEPSEEK_API_KEY:
        return JSONResponse(
            {"error": "DEEPSEEK_API_KEY non configurata"}, status_code=500
        )
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    try:
        resp = requests.get(ping_url, headers=headers, timeout=DEEPSEEK_TIMEOUT)
        return JSONResponse({"status": resp.status_code, "body": resp.text})
    except Exception as exc:
        logger.error("Errore pingando DeepSeek: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)


app.mount("/static", StaticFiles(directory="static"), name="static")

CONVERSATIONS: dict[str, ConversationBufferMemory] = {}

FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback.json")
FEEDBACK_STORAGE: list[dict] = []
FEEDBACK_LOCK = asyncio.Lock()


def _load_feedback_from_disk() -> list[dict]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as feedback_file:
            data = json.load(feedback_file)
            if isinstance(data, list):
                return data
            logger.warning(
                "Formato feedback non valido in %s: attesa lista, trovato %s",
                FEEDBACK_FILE,
                type(data).__name__,
            )
    except Exception as exc:
        logger.warning("Impossibile caricare i feedback esistenti: %s", exc)
    return []


def _save_feedback_to_disk(entries: list[dict]) -> None:
    tmp_path = f"{FEEDBACK_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as tmp_file:
        json.dump(entries, tmp_file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, FEEDBACK_FILE)


FEEDBACK_STORAGE.extend(_load_feedback_from_disk())

llm = None
retriever = None
embeddings = None
INIT_ERROR = None


class AskRequest(BaseModel):
    query: str
    agent_id: Optional[int] = None
    agent: Optional[str] = None
    session_id: Optional[str] = "default"
    include_image: Optional[bool] = True


class FeedbackRequest(BaseModel):
    session_id: str
    agent_id: int
    rating: conint(ge=1, le=5)
    commento: Optional[str] = None

# Provider di default: DeepSeek, in linea con README e script ausiliari
_provider_env = os.getenv("LLM_PROVIDER")
if not _provider_env or not _provider_env.strip():
    LLM_PROVIDER = "deepseek"
else:
    LLM_PROVIDER = _provider_env.strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    OPENAI_API_KEY = re.sub(r"\s+", "", OPENAI_API_KEY or "")

# Consente di sovrascrivere l'endpoint OpenAI, utile per ambienti personalizzati.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = re.sub(r"\s+", "", DEEPSEEK_API_KEY or "")
# Normalize base URL and derive API base with '/v1'
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
if DEEPSEEK_BASE_URL.rstrip('/').endswith('/v1'):
    DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL.rstrip('/')
    DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL[:-3]
DEEPSEEK_BASE_URL = DEEPSEEK_BASE_URL.rstrip('/')
DEEPSEEK_API_BASE = f"{DEEPSEEK_BASE_URL}/v1"
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "10"))
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
ENABLE_IMAGE_SEARCH = os.getenv("ENABLE_IMAGE_SEARCH", "true").lower() == "true"

EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "").strip().lower()

if OPENAI_API_KEY and not DEEPSEEK_API_KEY:
    LLM_PROVIDER = "openai"

if LLM_PROVIDER not in {"openai", "deepseek"}:
    raise Exception("LLM_PROVIDER deve essere 'openai' o 'deepseek'")

if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise Exception(
        "Variabile d'ambiente OPENAI_API_KEY mancante. "
        "Imposta OPENAI_API_KEY e LLM_PROVIDER=openai oppure "
        "fornisci DEEPSEEK_API_KEY e LLM_PROVIDER=deepseek."
    )
if LLM_PROVIDER == "deepseek" and not DEEPSEEK_API_KEY:
    raise Exception(
        "Variabile d'ambiente DEEPSEEK_API_KEY mancante. "
        "Imposta DEEPSEEK_API_KEY e LLM_PROVIDER=deepseek."
    )

VECTORDB_PATH = "vectordb/"
if not os.path.isdir(VECTORDB_PATH):
    INIT_ERROR = (
        f"Directory '{VECTORDB_PATH}' non trovata. Ricrea o committa l'indice FAISS."
    )
BASE_INSTRUCTION = (
    "Rispondi sempre in modo chiaro, tecnico, e senza ironia. "
    "Non aggiungere battute, frasi umoristiche o riferimenti surreali. Concentrati solo sulla risoluzione del problema. "
    "Se rilevi termini tecnici, aggiungi note a piè di pagina numerate con spiegazioni sintetiche. Se opportuno, includi un'immagine rilevante tramite Bing. "
    "Termina ogni risposta con una domanda mirata per approfondire la richiesta dell'utente."
)

if not INIT_ERROR:
    try:
        if LLM_PROVIDER == "openai":
            ping_url = f"{OPENAI_BASE_URL.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        else:  # deepseek
            ping_url = f"{DEEPSEEK_API_BASE}/models"
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
        requests.get(ping_url, headers=headers, timeout=DEEPSEEK_TIMEOUT).raise_for_status()

        if EMBEDDINGS_PROVIDER == "openai" and OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            )
        elif EMBEDDINGS_PROVIDER == "huggingface":
            model_name = os.getenv(
                "HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
        else:
            # se usi DeepSeek come LLM, carica l'indice FAISS senza ridefinire embedding
            embeddings = None

        db = FAISS.load_local(
            VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 5})

        if LLM_PROVIDER == "openai":
            # Default model can be overridden via OPENAI_MODEL and must exist in your account
            llm = ChatOpenAI(
                model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=0,
                openai_api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
            )
        else:  # deepseek
            llm = ChatDeepSeek(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_API_BASE,
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                temperature=0,
            )

        logger.info("✅ Ambiente base inizializzato correttamente.")
    except (openai.APIConnectionError, requests.exceptions.RequestException) as exc:
        provider_name = "OpenAI" if LLM_PROVIDER == "openai" else "DeepSeek"
        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            body = exc.response.text
            logger.error("Errore HTTP %s: %s", status, body)
            if status == 401:
                INIT_ERROR = (
                    f"Chiave API {provider_name} non valida (status {status}: {body})"
                )
            else:
                INIT_ERROR = (
                    f"Errore HTTP {status} dall'API {provider_name}: {body}"
                )
        else:
            INIT_ERROR = (
                f"Impossibile contattare l'API {provider_name}; verifica rete o chiave"
            )
            logger.error("Errore contattando l'API %s: %s", provider_name, exc)
        logger.debug(traceback.format_exc())
        logger.error(INIT_ERROR)
    except Exception as exc:
        INIT_ERROR = f"Errore durante l'inizializzazione della pipeline RAG: {exc}"
        logger.exception("❌ Errore durante l'inizializzazione della pipeline RAG:")

TOOLTIPS = {
    "filtro": "Componente da pulire regolarmente per evitare intasamenti e cattivi odori.",
    "filtri": "Componenti da pulire regolarmente per evitare intasamenti e cattivi odori.",
    "scarico": "Il sistema che espelle l'acqua dalla lavatrice o lavastoviglie.",
    "pompa": "Dispositivo che serve a espellere l'acqua dall’elettrodomestico.",
    "motore": "Cuore del funzionamento meccanico, può essere inverter o tradizionale.",
    "pressostato": "Dispositivo che misura la pressione dell’aria per determinare il livello dell’acqua; se guasto può bloccare carico, centrifuga o apertura dell’oblò.",
    "elettrovalvola": "Valvola elettrica che apre o chiude l’ingresso dell’acqua; bobine interrotte o bloccate impediscono il riempimento.",
    "resistenza": "Serve a riscaldare l’acqua nei cicli di lavaggio.",
    "guarnizione": "Elemento in gomma per prevenire perdite di acqua.",
    "scheda elettronica": "Il cervello dell’elettrodomestico: gestisce tutte le funzioni.",
    "errore": "Indicazione di guasto tramite codice alfanumerico.",
    "codice errore": "Sigla (es: E10, F06) che indica un malfunzionamento specifico.",
    "codici errore": "Serie di sigle usate per indicare malfunzionamenti tecnici.",
    "tastiera": "Interfaccia utente: pulsanti e manopole.",
    "programma": "Ciclo di lavaggio o asciugatura selezionato dall’utente.",
    "inverter": "Tipo di motore elettronico a basso consumo.",
    "cuscinetto": "Supporto a sfere che permette al cestello di ruotare; se usurato produce rumori metallici e può causare perdite.",
    "cuscinetti": "Supporti a sfere che permettono al cestello di ruotare; se usurati producono rumori metallici e possono causare perdite.",
    "tramoggia": "Convogliatore che porta acqua e detersivo dalla vaschetta alla vasca; se ostruita impedisce il carico dell’acqua.",
    "ntc": "Sensore di temperatura a coefficiente negativo, 20–30 kΩ a temperatura ambiente; valori fuori range indicano guasto.",
    "pompa di scarico": "Motore che evacua l’acqua dalla vasca verso lo scarico; ostruzioni o guasti bloccano scarico e centrifuga.",
    "ammortizzatore": "Elemento che assorbe le vibrazioni del gruppo vasca durante la centrifuga; se usurato la lavatrice si muove.",
    "ammortizzatori": "Elementi che assorbono le vibrazioni del gruppo vasca durante la centrifuga; se usurati la lavatrice si muove.",
    "contrappeso": "Peso in cemento o ghisa fissato alla vasca per bilanciare il cestello; se allentato provoca colpi in centrifuga.",
    "contrappesi": "Pesi in cemento o ghisa fissati alla vasca per bilanciare il cestello; se allentati provocano colpi in centrifuga.",
    "tachimetro": "Sensore montato sul motore che misura la velocità di rotazione; un guasto impedisce la corretta modulazione della centrifuga.",
    "tachimetrica": "Sensore montato sul motore che misura la velocità di rotazione; un guasto impedisce la corretta modulazione della centrifuga.",
    "fermo di trasporto": "Bullone che blocca il cestello durante il trasporto; se non rimosso causa vibrazioni e rumori.",
    "fermi di trasporto": "Bulloni che bloccano il cestello durante il trasporto; se non rimossi causano vibrazioni e rumori.",
    "contatti elettrici": "Punti di connessione che possono ossidarsi e interrompere il circuito.",
    "tubo di carico": "Condotto che immette l’acqua nell’elettrodomestico.",
    "tubo di scarico": "Condotto che espelle l’acqua dall’elettrodomestico.",
    "guarnizione oblò": "Anello di gomma che assicura la tenuta dello sportello.",
}

# Elenco degli agenti disponibili nel progetto
AGENTS = [
    {
        "id": 1,
        "nome": "Gustav",
        "descrizione": (
            "Tecnico esperto nella riparazione degli elettrodomestici. "
            "Guida l'utente con domande mirate e spiegazioni concise "
            "per una diagnosi efficace."
        ),
    },
    {
        "id": 2,
        "nome": "Yomo",
        "descrizione": (
            "Amica esperta in prodotti per la cura degli elettrodomestici. "
            "Suggerisce soluzioni pratiche e performanti per la manutenzione."
        ),
    },
    {
        "id": 3,
        "nome": "Jenna",
        "descrizione": (
            "Assistente per utilizzare al meglio gli elettrodomestici. "
            "Offre consigli pratici e curiosità per ottimizzare l'uso."
        ),
    },
    {
        "id": 4,
        "nome": "Liutprando",
        "descrizione": (
            "Consulente per la scelta degli elettrodomestici perfetti. "
            "Propone modelli su misura analizzando caratteristiche tecniche."
        ),
    },
    {
        "id": 5,
        "nome": "Manutentore interno",
        "descrizione": "Gestione debug e problematiche",
    },
]

# Brevi presentazioni per ciascun agente
AGENT_INTROS = {
    1: (
        "Gustav, il tecnico esperto nella riparazione degli elettrodomestici. "
        "Sono qui per aiutarti a diagnosticare rapidamente ogni guasto."
    ),
    2: (
        "Yomo, la tua amica esperta in prodotti per la cura degli elettrodomestici. "
        "Posso consigliarti soluzioni pratiche per la manutenzione quotidiana."
    ),
    3: (
        "Jenna, l'assistente per utilizzare al meglio i tuoi elettrodomestici. "
        "Ti svelo trucchi e strategie per ottenere sempre risultati eccellenti."
    ),
    4: (
        "Liutprando, il tuo consulente nella scelta degli elettrodomestici perfetti per te. "
        "Ti aiuto a confrontare modelli e caratteristiche tecniche."
    ),
    5: (
        "Manutentore interno. "
        "Gestisco il debug e ogni problematica tecnica dei tuoi apparecchi."
    ),
}

# Prompt personalizzati per ciascun agente
_AGENT_PROMPTS_DICT = {
    1: (
        "Sei Gustav, il tecnico esperto nella riparazione degli elettrodomestici. "
        "Inizia ogni risposta con 'Gustav, il tecnico esperto nella riparazione degli elettrodomestici.' "
        "Guida l'utente attraverso un processo strutturato di diagnosi e risoluzione problemi, "
        "ponendo domande mirate e offrendo spiegazioni tecniche chiare e concise. "
        "Cerca attivamente il contesto necessario per una diagnosi efficace. "
        "Non fare riferimento a passaggi o istruzioni precedenti se non li hai già forniti nella conversazione: quando servono, elencali esplicitamente. "
        + BASE_INSTRUCTION
    ),
    2: (
        "Sei Yomo, la tua amica esperta in prodotti per la cura degli elettrodomestici. "
        "Inizia ogni risposta con 'Yomo, la tua amica esperta in prodotti per la cura degli elettrodomestici.' "
        "Suggerisci con tono amichevole i prodotti migliori per la pulizia, manutenzione e ottimizzazione "
        "degli elettrodomestici. Offri soluzioni pratiche e performanti, adattate alle esigenze quotidiane del cliente. "
        + BASE_INSTRUCTION
    ),
    3: (
        "Sei Jenna, l'assistente per utilizzare al meglio i tuoi elettrodomestici. "
        "Inizia ogni risposta con 'Jenna, l'assistente per utilizzare al meglio i tuoi elettrodomestici.' "
        "Suggerisci trucchi, strategie e curiosità utili per ottimizzare l'uso degli elettrodomestici. "
        "Offri consigli pratici per migliorare i risultati, mantenendo un tono leggero, positivo e informativo. "
        + BASE_INSTRUCTION
    ),
    4: (
        "Sei Liutprando, il tuo consulente nella scelta degli elettrodomestici perfetti per te. "
        "Inizia ogni risposta con 'Liutprando, il tuo consulente nella scelta degli elettrodomestici perfetti per te.' "
        "Agisci come un commesso esperto, facendo domande per comprendere le esigenze dell'utente e "
        "fornendo informazioni dettagliate su dimensioni, classi energetiche e performance. "
        "Proponi gli elettrodomestici più adatti alle specifiche necessità del cliente. "
        + BASE_INSTRUCTION
    ),
    5: (
        "Sei il Manutentore interno, addetto al debug e alla gestione delle problematiche. "
        "Inizia ogni risposta con 'Manutentore interno'. "
        "Fornisci indicazioni puntuali per la risoluzione problemi e il debug. "
        + BASE_INSTRUCTION
    ),
}

AGENT_PROMPTS = MappingProxyType(_AGENT_PROMPTS_DICT)


def build_rag(system_instruction: str) -> RetrievalQA:
    """Crea una catena RAG con il prompt fornito."""

    question_prompt = PromptTemplate(
        template=(
            f"{system_instruction}\nContesto:\n{{context}}\n\nDomanda: {{question}}"
        ),
        input_variables=["context", "question"],
    )

    refine_prompt = PromptTemplate(
        template=(
            f"{system_instruction}\n{{existing_answer}}\n\nContesto aggiuntivo:\n{{context}}\n\nDomanda: {{question}}"
        ),
        input_variables=["existing_answer", "context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever,
        chain_type_kwargs={
            "question_prompt": question_prompt,
            "refine_prompt": refine_prompt,
            "document_variable_name": "context",
        },
    )

# Costruisce le catene RAG per ogni agente all'avvio dell'app in modo sicuro
RAG_CHAINS: dict[int, RetrievalQA] = {}
if not INIT_ERROR:
    try:
        RAG_CHAINS = {
            agent_id: build_rag(prompt) for agent_id, prompt in AGENT_PROMPTS.items()
        }
    except Exception as exc:
        RAG_CHAINS = {}
        logger.warning(
            "Impossibile costruire le catene RAG, uso fallback: %s", exc
        )

def applica_tooltip(testo: str) -> str:
    """Sostituisce i tooltip inline con note a piè di pagina numerate."""

    footnotes: list[str] = []
    indice_per_termine: dict[str, int] = {}

    # Costruisce un'unica regex che intercetta tutte le chiavi del dizionario.
    chiavi_ordinate = sorted(TOOLTIPS.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<![\\w>])(" + "|".join(map(re.escape, chiavi_ordinate)) + r")(?![\\w<])",
        re.IGNORECASE,
    )

    def _sostituisci(match):
        termine = match.group(0)
        chiave = termine.lower()
        spiegazione = TOOLTIPS.get(chiave, "")

        if chiave not in indice_per_termine:
            indice = len(indice_per_termine) + 1
            indice_per_termine[chiave] = indice
            footnotes.append(
                f'<li id="footnote-{indice}">{spiegazione} '
                f'<a href="#ref-{indice}">↩</a></li>'
            )
        else:
            indice = indice_per_termine[chiave]

        return (
            f"{termine}<sup id=\"ref-{indice}\">"
            f"<a href=\"#footnote-{indice}\">[{indice}]</a></sup>"
        )

    testo = pattern.sub(_sostituisci, testo)

    if footnotes:
        testo += "<hr /><ol class=\"footnotes\">" + "".join(footnotes) + "</ol>"

    return testo


async def cerca_immagine_bing(query: str, image_requested: bool = True) -> str:
    if not image_requested or not ENABLE_IMAGE_SEARCH or not BING_SEARCH_API_KEY:
        return ""

    def _search() -> str:
        headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
        params = {"q": query, "count": 1, "imageType": "Photo"}
        response = requests.get(
            "https://api.bing.microsoft.com/v7.0/images/search",
            headers=headers,
            params=params,
        )
        try:
            results = response.json()
            return results["value"][0]["contentUrl"] if results["value"] else ""
        except Exception:
            return ""

    return await asyncio.to_thread(_search)


def classify_query(question: str) -> int:
    q = question.lower()
    if any(
        w in q
        for w in [
            "errore",
            "guasto",
            "codice",
            "non funziona",
            "pompa",
            "scheda",
            "motore",
        ]
    ):
        return 1  # Gustav
    if any(w in q for w in ["pulizia", "manutenzione", "prodotto", "detergente"]):
        return 2  # Yomo
    if any(w in q for w in ["come usare", "consiglio d’uso", "trucchi", "ottimizzare"]):
        return 3  # Jenna
    if any(w in q for w in ["acquistare", "modello", "classe energetica"]):
        return 4  # Liutprando
    if any(w in q for w in ["debug", "diagnosi avanzata"]):
        return 5  # Manutentore interno
    return 1  # default a Gustav


@app.post("/ask")
async def ask_question(payload: AskRequest, request: Request):
    if INIT_ERROR:
        return JSONResponse(status_code=500, content={"error": INIT_ERROR})
    image_task = None
    try:
        user_question = payload.query.strip()
        session_id = payload.session_id or "default"
        memory = CONVERSATIONS.setdefault(
            session_id, ConversationBufferMemory(return_messages=False)
        )

        if not user_question:
            raise HTTPException(
                status_code=422, detail="Inserisci il campo 'query' nel JSON"
            )

        # Recupera l'id dell'agente, accettando sia 'agent_id' che 'agent'
        agent_raw = payload.agent_id or payload.agent
        if agent_raw is None:
            agent_id = classify_query(user_question)
        else:
            try:
                candidate = int(agent_raw)
                if candidate not in AGENT_PROMPTS:
                    raise ValueError()
                agent_id = candidate
            except (TypeError, ValueError):
                if isinstance(agent_raw, str):
                    name = agent_raw.strip().lower()
                    match = next(
                        (a["id"] for a in AGENTS if a["nome"].lower() == name), None
                    )
                    if match is not None:
                        agent_id = match
                    else:
                        raise HTTPException(
                            status_code=422,
                            detail={"error": "Invalid agent", "agenti": AGENTS},
                        )
                else:
                    raise HTTPException(
                        status_code=422,
                        detail={"error": "Invalid agent", "agenti": AGENTS},
                    )

        include_image = bool(payload.include_image)
        image_task = asyncio.create_task(
            cerca_immagine_bing(user_question, include_image)
        )

        logger.info(f"▶️ Ricevuta query: {user_question!r} per agente {agent_id}")

        # Gestisce la richiesta di introduzione senza invocare la RAG
        if user_question.lower() == "introduzione":
            answer = AGENT_INTROS[agent_id]
        else:
            # Esempio: Jenna non deve usare la RAG se la domanda è fuori ambito
            if agent_id == 3 and any(
                term in user_question.lower()
                for term in [
                    "errore",
                    "pompa",
                    "guasto",
                    "non funziona",
                    "codice",
                    "sostituire",
                ]
            ):
                answer = (
                    "Jenna, l'assistente per utilizzare al meglio i tuoi elettrodomestici. "
                    "Mi occupo di consigli sull'uso quotidiano, non di problemi tecnici. "
                    "Per assistenza su guasti o riparazioni, chiedi a Gustav, il tecnico esperto."
                )
            else:
                rag = RAG_CHAINS.get(agent_id)
                if rag:
                    try:
                        question_with_context = (
                            f"{memory.buffer}\nUtente: {user_question}" if memory.buffer else user_question
                        )
                        answer = await rag.arun(question_with_context)
                    except AssertionError:
                        await image_task
                        msg = (
                            "Indice FAISS non compatibile. Ricostruisci 'vectordb/' con lo stesso modello di embedding."
                        )
                        return JSONResponse(status_code=500, content={"error": msg})
                    except Exception:
                        logger.warning("RAG fallita, uso fallback")
                        context_section = (
                            f"\nContesto conversazione:\n{memory.buffer}" if memory.buffer else ""
                        )
                        fallback_prompt = (
                            f"{AGENT_PROMPTS[agent_id]}{context_section}\nDomanda: {user_question}"
                        )
                        answer = await llm.apredict(fallback_prompt)
                else:
                    logger.warning("Catena RAG assente, uso fallback")
                    context_section = (
                        f"\nContesto conversazione:\n{memory.buffer}" if memory.buffer else ""
                    )
                    fallback_prompt = (
                        f"{AGENT_PROMPTS[agent_id]}{context_section}\nDomanda: {user_question}"
                    )
                    answer = await llm.apredict(fallback_prompt)

        # Aggiorna la memoria con l'interazione corrente
        memory.chat_memory.add_user_message(user_question)
        memory.chat_memory.add_ai_message(answer)

        image_url = await image_task
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
    finally:
        if image_task is not None and not image_task.done():
            await image_task


@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="session_id obbligatorio")

    if payload.agent_id not in AGENT_PROMPTS:
        raise HTTPException(
            status_code=422,
            detail={"error": "Invalid agent", "agenti": AGENTS},
        )

    commento = (payload.commento or "").strip()
    entry = {
        "session_id": session_id,
        "agent_id": payload.agent_id,
        "rating": payload.rating,
        "commento": commento or None,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    async with FEEDBACK_LOCK:
        FEEDBACK_STORAGE.append(entry)
        try:
            await asyncio.to_thread(_save_feedback_to_disk, FEEDBACK_STORAGE)
        except Exception as exc:
            FEEDBACK_STORAGE.pop()
            logger.error("Errore durante il salvataggio del feedback: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Impossibile salvare il feedback in questo momento.",
            ) from exc

    logger.info(
        "💬 Feedback registrato per sessione %s (agente %s, rating %s)",
        session_id,
        payload.agent_id,
        payload.rating,
    )
    return {"status": "ok"}


@app.get("/health")
async def health():
    if INIT_ERROR:
        return {"status": "error", "detail": INIT_ERROR}
    return {"status": "ok"}


@app.get("/debug/ping")
async def debug_ping():
    try:
        r = requests.get(
            f"{DEEPSEEK_API_BASE}/models",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            timeout=5,
        )
        return {"status": r.status_code, "body": r.json()}
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/agents")
async def list_agents():
    """Restituisce l'elenco degli agenti configurati."""
    return {"agenti": AGENTS}
