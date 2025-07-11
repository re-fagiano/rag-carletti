import requests

# URL della tua API FastAPI
url = "http://127.0.0.1:8000/ask"

print("🤖 Assistente RAG attivo! Scrivi una domanda (digita 'esci' per uscire)\n")

while True:
    domanda = input("❓ Tu: ").strip()

    if not domanda:
        continue

    if domanda.lower() in ["esci", "exit", "quit", "q"]:
        print("👋 Fine della sessione. A presto!")
        break

    try:
        response = requests.post(url, json={"query": domanda})
        if response.status_code == 200:
            print("🤖 Bot:", response.json().get("risposta", "[Nessuna risposta]"))
        else:
            print(f"❌ Errore {response.status_code}: {response.text}")
    except Exception as e:
        print("⚠️ Errore di connessione:", e)
