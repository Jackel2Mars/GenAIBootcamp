import json
from pathlib import Path
from collections import Counter
from ctransformers import AutoModelForCausalLM
from tqdm import tqdm

# === CONFIG ===
INPUT_FILE = "extracted_traits.json"  # ou "annotated_traits.json"
OUTPUT_FILE = "annotated_traits_forced.json"
OUTPUT_DIR = Path("styles_traits_forced")
MAX_TOKENS = 32

STYLES = {
    "Traditional": "Classic and romantic gestures, gallant behavior, references to traditional dating or customs.",
    "Playful": "Humorous, teasing, light-hearted banter, enjoys jokes and playful provocation.",
    "Physical": "Focuses on touch, appearance, sensuality, or tactile flirtation.",
    "Sincere": "Authentic, emotionally open, shows vulnerability, expresses deep feelings.",
    "Polite": "Courteous, respectful, formal, avoids overly familiar tone."
}

# === Detect GPU/device ===
def detect_system_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"

device = detect_system_device()
print(f"[INFO] Using device: {device}")

# === Load model ===
model = AutoModelForCausalLM.from_pretrained(
    "/Users/user/GenAIBootcamp/FinalProject/DataEnginnering/models/mistral",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50 if device != "cpu" else 0
)

# === Load dataset ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# data doit être une liste de dicts avec au moins extracted_traits et style
if not isinstance(data, list):
    raise ValueError("Expected a list of dicts in input file")

# === Collect unique traits ===
all_traits = []
for entry in data:
    traits = entry.get("extracted_traits", [])
    if isinstance(traits, list):
        all_traits.extend([t.strip() for t in traits if t.strip()])

freq = Counter(all_traits)
unique_traits_sorted = sorted(freq.keys(), key=lambda t: freq[t])  # du moins fréquent au plus fréquent

print(f"[INFO] Found {len(unique_traits_sorted)} unique traits.")

# === Classification of unique traits ===
trait_to_style = {}
for trait in tqdm(unique_traits_sorted, desc="Classifying unique traits"):
    prompt = (
        "You are a linguist and social psychologist.\n"
        "Classify the following personality or flirting trait into EXACTLY ONE of these flirting styles.\n"
        "None is forbidden. If unsure, choose the closest style.\n"
        f"{json.dumps(STYLES, indent=2)}\n\n"
        f"Trait: {trait}\n"
        "Return ONLY the style name exactly as written above."
    )
    raw_output = model(prompt, max_new_tokens=MAX_TOKENS, temperature=0.0).strip()
    style = raw_output.split()[0] if raw_output else None
    if style not in STYLES:
        # fallback: choose the style whose name appears in output
        for s in STYLES:
            if s.lower() in raw_output.lower():
                style = s
                break
        if style not in STYLES:
            style = "Sincere"  # fallback final (jamais None)
    trait_to_style[trait] = style

print(f"[INFO] Mapping completed. Sample: {list(trait_to_style.items())[:10]}")

# === Propagate classification to all conversations ===
annotated = []
for entry in data:
    traits = entry.get("extracted_traits", [])
    if isinstance(traits, list):
        # choisir le style du premier trait mappé (ou garder celui existant si déjà correct)
        styles_from_traits = [trait_to_style.get(t) for t in traits if t in trait_to_style]
        chosen_style = None
        if styles_from_traits:
            chosen_style = styles_from_traits[0]  # premier trait → style dominant
        # si style vide ou None → assigner
        if not entry.get("style") or entry["style"] == "None":
            entry["style"] = chosen_style if chosen_style else "Sincere"
    annotated.append(entry)

# === Save annotated dataset ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(annotated, f, indent=2, ensure_ascii=False)

# === Save per-style files ===
OUTPUT_DIR.mkdir(exist_ok=True)
for style in STYLES:
    style_data = [x for x in annotated if x.get("style") == style]
    with open(OUTPUT_DIR / f"{style.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(style_data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Saved annotated dataset to {OUTPUT_FILE}")
print(f"[INFO] Saved per-style datasets to {OUTPUT_DIR}/")
