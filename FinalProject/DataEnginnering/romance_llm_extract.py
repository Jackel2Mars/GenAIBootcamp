import json
import requests
from tqdm import tqdm

# 📍 Fichiers d’entrée/sortie
INPUT_FILE = "romance_conversations.jsonl"
OUTPUT_FILE = "filtered_romantic_conversations.jsonl"

# 🔌 Configuration d'Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # ou "phi3" selon ton besoin

# 💬 Fonction d’évaluation via LLM
DEBUG_FILE = "debug.log"

def is_romantic_convo(convo):
    text = "\n".join([line["text"] for line in convo["lines"]])
    prompt = f"""
Classify the following conversation. Does it contain or suggest an explicit romantic or flirtatious intent (e.g. dating proposals, compliments with romantic undertones, obvious flirting)?

Do not infer romance from vague emotions or subtle tone. Only respond "yes" if romantic or flirtatious intent is clearly expressed.

Respond only with: yes or no.

Conversation:
{text}
"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        })

        raw_answer = response.json()["response"].strip()

        # 💾 Ajout du log
        with open(DEBUG_FILE, "a", encoding="utf-8") as log:
            log.write("===")
            log.write("\n")
            log.write("TEXT:\n")
            log.write(text[:500] + "...\n")
            log.write("RESPONSE:\n")
            log.write(raw_answer + "\n\n")

        return raw_answer.lower().startswith("yes")

    except Exception as e:
        print(f"⚠️ Erreur avec une conversation : {e}")
        return False

# 🚀 Fonction principale de filtrage
def filter_romantic_conversations():
    print("🔍 Début du filtrage romantique avec LLM...")

    kept = 0
    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Traitement des conversations"):
            try:
                convo = json.loads(line)
                total += 1
                if len(convo.get("lines", [])) < 2:
                    continue
                if is_romantic_convo(convo):
                    json.dump(convo, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    kept += 1
            except json.JSONDecodeError:
                continue

    print(f"✅ Conversations filtrées : {kept} sur {total}")

if __name__ == "__main__":
    filter_romantic_conversations()
