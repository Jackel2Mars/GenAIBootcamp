#!/usr/bin/env python3
# extract_traits.py
import json
from collections import Counter, defaultdict
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from ctransformers import AutoModelForCausalLM
import os

# -------------------------
# Config utilisateur
MODEL_DIR = "/Users/user/GenAIBootcamp/FinalProject/DataEnginnering/models/mistral"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_TYPE = "mistral"
MODEL_GPU_LAYERS = 50  # ajuste selon ta config (0 = CPU total)
BATCH_SIZE = 16
OUTPUT_JSON = "extracted_traits.json"
HIST_PNG = "traits_histogram.png"
# -------------------------

def detect_system_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS (Apple Silicon)
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    except Exception:
        return "cpu (torch not installed)"

system_device = detect_system_device()

# Charger le modèle ctransformers
print(f"[init] Loading model from {MODEL_DIR}/{MODEL_FILE} (model_type={MODEL_TYPE})")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    model_file=MODEL_FILE,
    model_type=MODEL_TYPE,
    gpu_layers=MODEL_GPU_LAYERS
)
print(f"[init] Model loaded. system_device={system_device}, model_gpu_layers={MODEL_GPU_LAYERS}")

def ask(prompt, max_tokens=512, temperature=0.2):
    """
    Appel au modèle. Affiche l'info device à CHAQUE appel (comme demandé).
    Retourne la string brute générée.
    """
    print(f"[ask] device={system_device} | model_gpu_layers={MODEL_GPU_LAYERS} | max_tokens={max_tokens}")
    out = model(prompt, max_new_tokens=max_tokens, temperature=temperature)
    # ctransformers peut retourner une string ou autre; normaliser en string
    if isinstance(out, (list, tuple)) and len(out) > 0:
        text = out[0] if isinstance(out[0], str) else str(out[0])
    elif isinstance(out, dict):
        # parfois dict avec 'content' ou 'generated_text'
        text = out.get("generated_text") or out.get("content") or json.dumps(out)
    else:
        text = str(out)
    return text

def extract_personality_traits_batch(batch_personalities):
    """
    batch_personalities: list of lists of personality strings
    returns: list of lists (extracted traits per entry)
    """
    results = []
    for personalities in batch_personalities:
        prompt = (
            "You are an expert psychologist specialized in extracting core behavioral personality traits "
            "from text descriptions of people.\n\n"
            "Given a list of personality descriptions, extract only the core personality traits, "
            "i.e. character-related qualities or stable behavioral tendencies.\n\n"
            "Do NOT extract interests, hobbies, professions, physical descriptions, preferences, or subjective opinions.\n\n"
            "Return your answer as a JSON list of simple lowercase strings, each string being a single trait.\n\n"
            f"List: {json.dumps(personalities, ensure_ascii=False)}"
        )
        output = ask(prompt)
        # output peut contenir du flux/texte additionnel ; on essaie d'extraire la première liste JSON visible
        traits = []
        try:
            # tenter un json.loads direct
            traits = json.loads(output)
        except Exception:
            # fallback: chercher le premier substring JSON (commençant par [ et se terminant par ])
            import re
            m = re.search(r"(\[.*\])", output, flags=re.S)
            if m:
                try:
                    traits = json.loads(m.group(1))
                except Exception:
                    traits = []
            else:
                traits = []

        # normaliser
        cleaned = []
        for t in traits if isinstance(traits, (list,tuple)) else []:
            if isinstance(t, str):
                tt = t.strip().lower()
                if tt:
                    cleaned.append(tt)
        results.append(cleaned)
        # debug léger
        if not cleaned:
            print(f"[warn] no traits extracted for input: {personalities}\n raw_output: {output[:400]}")
    return results

# ---- 1. Charger dataset et ajouter ID colonne proprement ----
print("[data] Loading dataset...")
ds = load_dataset("AlekseyKorshuk/synthetic-romantic-characters", split="train")
ds = ds.map(lambda example, idx: {"id": f"conv_{idx}"}, with_indices=True)
print(f"[data] Dataset loaded: {len(ds)} examples")

# ---- 2. Filtrer romance ----
romance_entries = [entry for entry in ds if any("romance" in c.lower() for c in entry.get("categories", []))]
print(f"[data] Romance entries: {len(romance_entries)}")

# ---- 3. Extraction traits (batch) ----
extracted_results = []  # list of dicts to save
romance_traits = set()

print("Extraction des traits depuis les entrées 'romance' (LLM)...")
for i in tqdm(range(0, len(romance_entries), BATCH_SIZE), desc="Extraction traits romance"):
    batch = romance_entries[i : i + BATCH_SIZE]
    batch_personalities = [entry.get("personalities", []) for entry in batch]
    extracted = extract_personality_traits_batch(batch_personalities)
    for entry, traits in zip(batch, extracted):
        extracted_results.append({
            "id": entry["id"],
            "movie_id": entry.get("movie_id"),
            "personalities": entry.get("personalities", []),
            "extracted_traits": traits
        })
        romance_traits.update(set(traits))

print(f"[result] Nombre de traits d'intérêt (romance): {len(romance_traits)}")

# Sauvegarde intermédiaire JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as fout:
    json.dump(extracted_results, fout, ensure_ascii=False, indent=2)
print(f"[io] Saved extracted results to {OUTPUT_JSON} ({len(extracted_results)} records)")

# ---- 4. Comptage global sur tout le dataset ----
freq_total = Counter()
freq_convos = defaultdict(set)

print("Comptage global (tous contexts) des traits romance...")
for entry in tqdm(ds, desc="Comptage global"):
    convo_id = entry["id"]
    personalities = [p.lower() for p in entry.get("personalities", [])]
    personalities_set = set(personalities)
    for trait in romance_traits:
        if trait in personalities_set:
            freq_total[trait] += 1
            freq_convos[trait].add(convo_id)

# Affichage rapide
print("\nTrait\tTotal Occurrences\t# Conversations")
for trait, total in freq_total.most_common(40):
    print(f"{trait}\t{total}\t{len(freq_convos[trait])}")

# Optionnel: inspect example convo ids pour un trait
example_trait = "adventurous"
print(f"\nExemples de convo ids pour '{example_trait}': {list(freq_convos[example_trait])[:10]} (total {len(freq_convos[example_trait])})")

# Top 20 traits
top_traits = freq_total.most_common(20)
if top_traits:
    traits, counts = zip(*top_traits)
else:
    traits, counts = [], []

plt.figure(figsize=(10, 6))
plt.bar(traits, counts)
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Personality Traits Extracted")
plt.xlabel("Trait")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(HIST_PNG)
plt.show()
print(f"[io] Histogram saved to {HIST_PNG}")

all_traits = list(freq_total.keys())

# 1. Longueur moyenne
avg_len = sum(len(t) for t in all_traits) / len(all_traits)
print(f"\nLongueur moyenne des traits: {avg_len:.2f} caractères")

# 2. Traits suspects
short_traits = [t for t in all_traits if len(t) < 3]
long_traits = [t for t in all_traits if len(t) > 30]
non_alpha_traits = [t for t in all_traits if not t.replace(" ", "").isalpha()]
print(f"Traits trop courts (<3): {len(short_traits)}")
print(f"Traits trop longs (>30): {len(long_traits)}")
print(f"Traits avec caractères non alphabétiques: {len(non_alpha_traits)}")

# 3. Taux d’unicité
total_occurrences = sum(freq_total.values())
unique_ratio = len(all_traits) / total_occurrences
print(f"Taux d'unicité: {unique_ratio:.2%}")