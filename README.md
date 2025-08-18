# RAG Carletti

Questo progetto fornisce un semplice esempio di Retrieval-Augmented Generation (RAG) basato su FastAPI e LangChain.
Viene fornito un dataset di documenti tecnici nel percorso `docs/` e un indice FAISS già costruito in `vectordb/`.

## Agenti del progetto

Nel file [AGENTS_INFO.md](AGENTS_INFO.md) sono elencati gli agenti disponibili e i loro ruoli:

| # | Nome | Descrizione |
|---|------|-------------|
| 1 | Gustav | Tecnico esperto nella riparazione degli elettrodomestici. Guida l'utente con domande mirate e spiegazioni concise per una diagnosi efficace. |
| 2 | Yomo | Amica esperta in prodotti per la cura degli elettrodomestici. Suggerisce soluzioni pratiche e performanti per la manutenzione. |
| 3 | Jenna | Assistente per utilizzare al meglio gli elettrodomestici. Offre consigli pratici e curiosità per ottimizzare l'uso. |
| 4 | Liutprando | Consulente per la scelta degli elettrodomestici perfetti. Propone modelli su misura analizzando caratteristiche tecniche. |
| 5 | Manutentore interno | Gestione debug e problematiche |

La stessa lista è disponibile via API con una richiesta `GET /agents`.

Per maggiori dettagli sui prompt utilizzati da ciascun agente consulta il file
[docs/AGENT_PROMPTS.md](docs/AGENT_PROMPTS.md). Le stesse istruzioni sono
implementate nel codice nella variabile `AGENT_PROMPTS` di `main.py`.

## Preparazione dell'ambiente

1. **Installazione delle dipendenze**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # per indicizzare anche i PDF
   pip install "langchain-community[pdf]"
   ```

2. **Variabili d'ambiente**
   Imposta almeno la chiave di OpenAI prima di avviare l'applicazione. È possibile
   specificare un modello diverso tramite la variabile `OPENAI_MODEL` (predefinito `gpt-5`):
   ```bash
   export OPENAI_API_KEY=<la tua chiave>
   export BING_SEARCH_API_KEY=<opzionale per immagini>
   export OPENAI_MODEL=<modello opzionale>
   export ENABLE_IMAGE_SEARCH=true  # disabilita con false
   ```

3. **Avvio dell'applicazione**
   ```bash
   uvicorn main:app --reload
   ```
   L'interfaccia web sarà disponibile su `http://127.0.0.1:8000`.

### Utilizzo tramite Docker

```bash
# build image
docker build -t rag-carletti .
# esegui l'app
docker run -p 8000:8000 -e OPENAI_API_KEY=<la tua chiave> \
    -e OPENAI_MODEL=<modello opzionale> rag-carletti
```

## Endpoint /ask
L'endpoint accetta il campo JSON `query` e, opzionalmente, `agent_id` (o `agent`) per scegliere quale agente deve rispondere. Se il parametro è assente verrà usato Gustav (id `1`). È possibile indicare l'id numerico o il nome dell'agente (non viene fatta distinzione tra maiuscole e minuscole). Se il valore non è riconosciuto l'API restituisce errore `422`. L'elenco completo degli agenti è consultabile con `GET /agents`.

La ricerca immagini tramite Bing può essere disabilitata globalmente impostando la variabile d'ambiente `ENABLE_IMAGE_SEARCH=false` oppure per singola richiesta passando `"include_image": false` nel payload.

Esempi di richiesta:
```bash
curl -X POST http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"query": "Perché la lavatrice non scarica?", "agent_id": 3}'

curl -X POST http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"query": "Consigli per la manutenzione", "agent": "yomo", "include_image": false}'
```

## Aggiornamento dell'indice dei documenti

Se modifichi o aggiungi file nella cartella `docs/` devi rigenerare la cartella `vectordb/` per riflettere i nuovi contenuti.

Puoi utilizzare uno dei seguenti script (richiedono entrambi la variabile d'ambiente `OPENAI_API_KEY` impostata):

```bash
python index_documents.py      # indicizza i soli file .txt
# oppure
python rebuild_vectordb.py
```

Al termine l'indice FAISS nella cartella `vectordb/` sarà aggiornato con le ultime modifiche.

Se necessario puoi includere anche i PDF passandone l'opzione `--include-pdf`
agli script sopra. Il testo verrà estratto e suddiviso in sezioni rilevanti
prima di calcolare le embedding; questa fase di parsing può richiedere
significativamente più tempo rispetto all'indicizzazione dei soli file di
testo.
