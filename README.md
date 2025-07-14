# RAG Carletti

Questo progetto fornisce un semplice esempio di Retrieval-Augmented Generation (RAG) basato su FastAPI e LangChain.
Viene fornito un dataset di documenti tecnici nel percorso `docs/` e un indice FAISS già costruito in `vectordb/`.

## Agenti del progetto

Nel file [AGENTS_INFO.md](AGENTS_INFO.md) sono elencati gli agenti disponibili e i loro ruoli:

| # | Nome | Descrizione |
|---|------|-------------|
| 1 | Gustav | Riparatore tecnico esperto di elettrodomestici |
| 2 | Yomo | Validatore e cercatore di informazioni |
| 3 | Jenna | Esperto di utilizzo degli elettrodomestici |
| 4 | Liutprando | Comico venditore esperto di elettrodomestici |
| 5 | Manutentore interno | Gestione debug e problematiche |

La stessa lista è disponibile via API con una richiesta `GET /agents`.

## Preparazione dell'ambiente

1. **Installazione delle dipendenze**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Variabili d'ambiente**
   Imposta almeno la chiave di OpenAI prima di avviare l'applicazione:
   ```bash
   export OPENAI_API_KEY=<la tua chiave>
   export BING_SEARCH_API_KEY=<opzionale per immagini>
   ```

3. **Avvio dell'applicazione**
   ```bash
   python main.py
   ```
   L'interfaccia web sarà disponibile su `http://127.0.0.1:8000`.

### Utilizzo tramite Docker

In alternativa è possibile avviare l'app tramite Docker:
main.py
+15
-0

@@ -73,50 +73,59 @@ try:
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

# Elenco degli agenti disponibili nel progetto
AGENTS = [
    {"id": 1, "nome": "Gustav", "descrizione": "Riparatore tecnico esperto di elettrodomestici"},
    {"id": 2, "nome": "Yomo", "descrizione": "Validatore e cercatore di informazioni"},
    {"id": 3, "nome": "Jenna", "descrizione": "Esperto di utilizzo degli elettrodomestici"},
    {"id": 4, "nome": "Liutprando", "descrizione": "Comico venditore esperto di elettrodomestici"},
    {"id": 5, "nome": "Manutentore interno", "descrizione": "Gestione debug e problematiche"},
]


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
@@ -133,25 +142,31 @@ async def ask_question(request: Request):
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


@app.get("/agents")
async def list_agents():
    """Restituisce l'elenco degli agenti configurati."""
    return {"agenti": AGENTS}
