import argparse
import uuid

import requests

# URL della tua API FastAPI
BASE_URL = "http://127.0.0.1:8000"
ASK_URL = f"{BASE_URL}/ask"
FEEDBACK_URL = f"{BASE_URL}/feedback"

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

session_id = str(uuid.uuid4())

print("🤖 Assistente RAG attivo! Scrivi una domanda (digita 'esci' per uscire)\n")
print(f"ℹ️  ID sessione: {session_id}\n")

while True:
    domanda = input("❓ Tu: ").strip()

    if not domanda:
        continue

    if domanda.lower() in ["esci", "exit", "quit", "q"]:
        print("👋 Fine della sessione. A presto!")
        break

    try:
        response = requests.post(
            ASK_URL,
            json={"query": domanda, "agent_id": agent_id, "session_id": session_id},
        )
        if response.status_code == 200:
            risposta = response.json().get("risposta", "[Nessuna risposta]")
            print("🤖 Bot:", risposta)
            while True:
                lascia = input("Vuoi lasciare un feedback? [s/N]: ").strip().lower()
                if lascia in {"", "n", "no"}:
                    break
                if lascia in {"s", "si", "sì"}:
                    rating = None
                    while rating is None:
                        voto = input("Valuta la risposta da 1 a 5: ").strip()
                        try:
                            valore = int(voto)
                            if 1 <= valore <= 5:
                                rating = valore
                            else:
                                print("Inserisci un numero tra 1 e 5.")
                        except ValueError:
                            print("Inserisci un numero valido.")
                    commento = input("Commento (opzionale): ").strip()
                    payload = {
                        "session_id": session_id,
                        "agent_id": agent_id,
                        "rating": rating,
                    }
                    if commento:
                        payload["commento"] = commento
                    try:
                        fb_resp = requests.post(FEEDBACK_URL, json=payload)
                        if fb_resp.status_code == 200:
                            print("✅ Feedback inviato, grazie!\n")
                        else:
                            print(
                                f"❌ Impossibile inviare il feedback ({fb_resp.status_code}): {fb_resp.text}"
                            )
                    except Exception as exc:
                        print("⚠️ Errore durante l'invio del feedback:", exc)
                    break
                else:
                    print("Risposta non riconosciuta. Digita 's' per sì o 'n' per no.")
        else:
            print(f"❌ Errore {response.status_code}: {response.text}")
    except Exception as e:
        print("⚠️ Errore di connessione:", e)
