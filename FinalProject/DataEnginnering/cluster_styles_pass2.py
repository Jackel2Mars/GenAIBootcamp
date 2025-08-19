import json
from pathlib import Path
from ctransformers import AutoModelForCausalLM
from tqdm import tqdm

# === CONFIG ===
INPUT_ANNOTATED = "annotated_traits.json"
OUTPUT_ANNOTATED = "annotated_traits_pass2_temp03.json"
OUTPUT_DIR = Path("styles_traits_pass2_temp03")
MAX_TOKENS = 64

STYLES = {
    "Traditional": "Classic and romantic gestures, gallant behavior, references to traditional dating or customs.",
    "Playful": "Humorous, teasing, light-hearted banter, enjoys jokes and playful provocation.",
    "Physical": "Focuses on touch, appearance, sensuality, or tactile flirtation.",
    "Sincere": "Authentic, emotionally open, shows vulnerability, expresses deep feelings.",
    "Polite": "Courteous, respectful, formal, avoids overly familiar tone."
}

EXAMPLES = [
    ("witty", "Playful"),
    ("courteous", "Polite"),
    ("romantic", "Traditional"),
    ("touch-oriented", "Physical"),
    ("empathetic", "Sincere"),
]

# === Detect device ===
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

system_device = detect_system_device()
print(f"[INFO] Using device: {system_device}")

# === Load model ===
model = AutoModelForCausalLM.from_pretrained(
    "/Users/user/GenAIBootcamp/FinalProject/DataEnginnering/models/mistral",
    model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50 if system_device != "cpu" else 0
)

# === Load annotated dataset ===
with open(INPUT_ANNOTATED, "r", encoding="utf-8") as f:
    annotated = json.load(f)

none_items = [x for x in annotated if x["style"] == "None"]
print(f"[INFO] Retraitement de {len(none_items)} traits classés 'None'.")

# === Reclassement permissif ===
for item in tqdm(none_items, desc="Reclassifying None traits"):
    examples_str = "\n".join([f"Trait: {t}\nStyle: {s}" for t, s in EXAMPLES])
    prompt = (
        "You are a linguist and social psychologist.\n"
        "Classify the following trait into EXACTLY ONE of these flirting styles, even if the match is partial:\n"
        f"{json.dumps(STYLES, indent=2)}\n\n"
        "Here are some examples:\n"
        f"{examples_str}\n\n"
        f"Trait: {item['trait']}\n"
        "Return only the style name exactly as written above."
    )
    raw_output = model(prompt, max_new_tokens=MAX_TOKENS, temperature=0.3).strip()
    style = raw_output.split()[0] if raw_output else "None"
    if style not in STYLES:
        style = "None"
    item["style"] = style

# === Sauvegarde du dataset fusionné ===
with open(OUTPUT_ANNOTATED, "w", encoding="utf-8") as f:
    json.dump(annotated, f, indent=2, ensure_ascii=False)

# === Sauvegarde par style ===
OUTPUT_DIR.mkdir(exist_ok=True)
for style in STYLES.keys():
    style_data = [x for x in annotated if x["style"] == style]
    with open(OUTPUT_DIR / f"{style.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(style_data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Pass2 terminée avec température 0.3. Nouveau fichier : {OUTPUT_ANNOTATED}")
