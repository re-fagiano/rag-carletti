import argparse
import httpx

# URL della tua API FastAPI
url = "http://127.0.0.1:8000/ask"

parser = argparse.ArgumentParser(description="Client CLI per interrogare l'API /ask")
parser.add_argument("--agent-id", "-a", type=int, help="ID dell'agente da utilizzare")
args = parser.parse_args()

agent_id = args.agent_id
if agent_id is None:
    scelta = input(
        "Seleziona l'agente (1=Gustav, 2=Yomo, 3=Jenna, 4=Liutprando, 5=Manutentore interno) [1]: "
    ).strip()
    try:
        agent_id = int(scelta or 1)
    except ValueError:
        agent_id = 1

print("🤖 Assistente RAG attivo! Scrivi una domanda (digita 'esci' per uscire)\n")

while True:
    domanda = input("❓ Tu: ").strip()

    if not domanda:
        continue

    if domanda.lower() in ["esci", "exit", "quit", "q"]:
        print("👋 Fine della sessione. A presto!")
        break

    try:
        response = httpx.post(url, json={"query": domanda, "agent_id": agent_id})
        if response.status_code == 200:
            print("🤖 Bot:", response.json().get("risposta", "[Nessuna risposta]"))
        else:
            print(f"❌ Errore {response.status_code}: {response.text}")
    except Exception as e:
        print("⚠️ Errore di connessione:", e)
