import json
from pathlib import Path
from ctransformers import AutoModelForCausalLM
from tqdm import tqdm

# === CONFIG ===
INPUT_FILE = "extracted_traits.json"
OUTPUT_ANNOTATED = "annotated_traits.json"
OUTPUT_DIR = Path("styles_traits")
OUTPUT_NONE = "annotated_none.json"
MAX_TOKENS = 32
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
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
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
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

traits_list = data if isinstance(data, list) else data.get("extracted_traits", [])

annotated = []
none_list = []

# === Classification séquentielle ===
for trait in tqdm(traits_list, desc="Classifying traits"):
    prompt = (
        "You are a linguist and social psychologist.\n"
        "Classify the following personality trait into EXACTLY ONE of these flirting styles:\n"
        f"{json.dumps(STYLES, indent=2)}\n\n"
        f"Trait: {trait}\n"
        "Rules:\n"
        "- Choose only one of the 5 styles.\n"
        "- If there is no clear match, answer exactly 'None'.\n"
        "- Output must be only the style name with no extra words.\n"
        "Examples:\n"
        "funny -> Playful\n"
        "romantic -> Traditional\n"
        "touchy -> Physical\n"
        "honest -> Sincere\n"
        "respectful -> Polite\n"
        "angry -> None\n"
        "\nAnswer:"
    )
    
    raw_output = model(prompt, max_new_tokens=MAX_TOKENS, temperature=0.0).strip()
    style = raw_output.split()[0] if raw_output else "None"
    if style not in STYLES:
        style = "None"
    
    annotated.append({"trait": trait, "style": style})
    if style == "None":
        none_list.append({"trait": trait, "style": style})

# === Save annotated dataset ===
with open(OUTPUT_ANNOTATED, "w", encoding="utf-8") as f:
    json.dump(annotated, f, indent=2, ensure_ascii=False)

# === Save None cases ===
with open(OUTPUT_NONE, "w", encoding="utf-8") as f:
    json.dump(none_list, f, indent=2, ensure_ascii=False)

# === Save per-style datasets ===
OUTPUT_DIR.mkdir(exist_ok=True)
for style in STYLES.keys():
    style_data = [x for x in annotated if x["style"] == style]
    with open(OUTPUT_DIR / f"{style.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(style_data, f, indent=2, ensure_ascii=False)

print(f"Saved annotated dataset to {OUTPUT_ANNOTATED}")
print(f"Saved per-style datasets to {OUTPUT_DIR}/")
print(f"Saved None cases to {OUTPUT_NONE}")