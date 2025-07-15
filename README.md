# RAG Carletti

Questo progetto fornisce un semplice esempio di Retrieval-Augmented Generation (RAG) basato su FastAPI e LangChain.
Viene fornito un dataset di documenti tecnici nel percorso `docs/` e un indice FAISS già costruito in `vectordb/`.

## Agenti del progetto

Nel file [AGENTS_INFO.md](AGENTS_INFO.md) sono elencati gli agenti disponibili e i loro ruoli:

| # | Nome | Descrizione |
|---|------|-------------|
| 1 | Gustav | Tecnico esperto nella riparazione degli elettrodomestici, guida la diagnosi con domande mirate |
| 2 | Yomo | Amica esperta in prodotti per la cura degli elettrodomestici |
| 3 | Jenna | Assistente che suggerisce trucchi e strategie per usare al meglio gli elettrodomestici |
| 4 | Liutprando | Consulente nella scelta degli elettrodomestici, esperto di caratteristiche tecniche |
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

```bash
# build image
docker build -t rag-carletti .
# esegui l'app
docker run -p 8000:8000 -e OPENAI_API_KEY=<la tua chiave> rag-carletti
```

## Endpoint /ask
L'endpoint accetta il campo JSON `query` e, opzionalmente, `agent_id` (o `agent`) per scegliere quale agente deve rispondere. Se il parametro è assente o non valido verrà usato Gustav (id `1`). L'elenco completo degli agenti è consultabile anche con `GET /agents`.

Esempio di richiesta:
```bash
curl -X POST http://localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"query": "Perché la lavatrice non scarica?", "agent_id": 3}'
```
