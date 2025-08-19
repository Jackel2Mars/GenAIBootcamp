import json
from pathlib import Path
from ctransformers import AutoModelForCausalLM
from tqdm import tqdm

# === CONFIG ===
INPUT_FILE = "extracted_traits.json"
OUTPUT_ANNOTATED = "annotated_traits.json"
OUTPUT_DIR = Path("styles_traits")
MAX_TOKENS = 64  # traits sont courts, pas besoin de beaucoup
STYLES = {
    "Traditional": "Classic and romantic gestures, gallant behavior, references to traditional dating or customs.",
    "Playful": "Humorous, teasing, light-hearted banter, enjoys jokes and playful provocation.",
    "Physical": "Focuses on touch, appearance, sensuality, or tactile flirtation.",
    "Sincere": "Authentic, emotionally open, shows vulnerability, expresses deep feelings.",
    "Polite": "Courteous, respectful, formal, avoids overly familiar tone."
}

# === Detect GPU / device ===
def detect_system_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    except Exception:
        return "cpu"

system_device = detect_system_device()
print(f"[INFO] Using device: {system_device}")

# === Load Mistral model ===
model = AutoModelForCausalLM.from_pretrained(
    "/Users/user/GenAIBootcamp/FinalProject/DataEnginnering/models/mistral",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50 if system_device != "cpu" else 0
)

# === Load traits ===
# === Load traits ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Si data est une liste directe de traits
traits_list = data if isinstance(data, list) else data.get("extracted_traits", [])

annotated = []

# === Classification ===
for trait in tqdm(traits_list, desc="Classifying traits"):
    prompt = (
        "You are a linguist and social psychologist.\n"
        "Classify the following trait into EXACTLY ONE of these flirting styles:\n"
        f"{json.dumps(STYLES, indent=2)}\n\n"
        f"Trait: {trait}\n"
        "Return only the name of the style exactly as written above, or 'None' if it doesn't match.\nAnswer:"
    )
    
    raw_output = model(prompt, max_new_tokens=MAX_TOKENS, temperature=0.0).strip()
    style = raw_output.split()[0] if raw_output else "None"
    if style not in STYLES:
        style = "None"
    
    annotated.append({"trait": trait, "style": style})

# === Save full annotated set ===
with open(OUTPUT_ANNOTATED, "w", encoding="utf-8") as f:
    json.dump(annotated, f, indent=2, ensure_ascii=False)

# === Save per-style datasets ===
OUTPUT_DIR.mkdir(exist_ok=True)
for style in STYLES.keys():
    style_data = [x for x in annotated if x["style"] == style]
    with open(OUTPUT_DIR / f"{style.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(style_data, f, indent=2, ensure_ascii=False)

print(f"Saved annotated dataset to {OUTPUT_ANNOTATED}")
print(f"Saved per-style datasets to {OUTPUT_DIR}/")
