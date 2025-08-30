# Usa Python ufficiale
FROM python:3.11-slim

# Crea cartella app
WORKDIR /app

# Copia tutto
COPY . .

# Installa i certificati
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

# Installa le dipendenze
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Espone la porta FastAPI
EXPOSE 8000

# Avvia FastAPI con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

