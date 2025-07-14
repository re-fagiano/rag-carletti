# RAG Carletti

Questo progetto fornisce un semplice esempio di Retrieval-Augmented Generation (RAG) basato su FastAPI e LangChain.
Viene fornito un dataset di documenti tecnici nel percorso `docs/` e un indice FAISS già costruito in `vectordb/`.

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
```bash
docker build -t rag-carletti .
docker run -p 8000:8000 \
    -e OPENAI_API_KEY=<la tua chiave> \
    -e BING_SEARCH_API_KEY=<opzionale> \
    rag-carletti
```

## Ricostruzione dell'indice FAISS

Se desideri rigenerare i vettori a partire dai file presenti in `docs/`, esegui:
```bash
python rebuild_vectordb.py
```
Questo script sovrascriverà il contenuto della cartella `vectordb/`.
