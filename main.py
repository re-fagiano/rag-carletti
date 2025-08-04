import logging
import traceback
import os
import re
import requests
from types import MappingProxyType
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# LangChain / OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
    raise Exception(
        f"Directory '{VECTORDB_PATH}' non trovata. Ricrea o committa l'indice FAISS."
    )

try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.load_local(
        VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY
    )

    BASE_INSTRUCTION = (
        "Rispondi sempre in modo chiaro, tecnico, e senza ironia. "
        "Non aggiungere battute, frasi umoristiche o riferimenti surreali. Concentrati solo sulla risoluzione del problema. "
        "Se rilevi termini tecnici, formattali con i tooltip. Se opportuno, includi un'immagine rilevante tramite Bing."
    )

    logger.info("✅ Ambiente base inizializzato correttamente.")
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
    "inverter": "Tipo di motore elettronico a basso consumo.",
    "cuscinetto": "Componente meccanico che consente al tamburo di girare senza attrito.",
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


def applica_tooltip(testo: str) -> str:
    for chiave, spiegazione in TOOLTIPS.items():
        pattern = r"(?<![\\w>])(" + re.escape(chiave) + r")(?![\\w<])"
        replacement = (
            r'<span class="tooltip">\1 <span class="info-icon">ⓘ</span>'
            r'<span class="tooltiptext">' + spiegazione + r"</span>"
            r"</span>"
        )
        testo = re.sub(pattern, replacement, testo, flags=re.IGNORECASE)
    return testo


def cerca_immagine_bing(query):
    if not BING_SEARCH_API_KEY:
        return ""
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
async def ask_question(request: Request):
    try:
        payload = await request.json()
        user_question = payload.get("query", "").strip()

        if not user_question:
            raise HTTPException(
                status_code=422, detail="Inserisci il campo 'query' nel JSON"
            )

        # Recupera l'id dell'agente, accettando sia 'agent_id' che 'agent'
        agent_raw = payload.get("agent_id") or payload.get("agent")
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
                rag = build_rag(AGENT_PROMPTS[agent_id])
                try:
                    answer = rag.run(user_question)
                except AssertionError:
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


@app.get("/agents")
async def list_agents():
    """Restituisce l'elenco degli agenti configurati."""
    return {"agenti": AGENTS}
