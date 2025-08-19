import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
from datasets import load_dataset
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# === CONFIG ===
DATASET_NAME = "AlekseyKorshuk/synthetic-romantic-characters"
CATEGORY_FILTER = "romance"
MIN_FREQ = 5
MODEL_PATH = "/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
STYLES = ["Traditional", "Playful", "Physical", "Sincere", "Polite"]

# Synonym mapping
TRAIT_SYNONYMS = {
    "playful": ["mischievous", "teasing", "cheeky", "flirty"],
    "sincere": ["genuine", "earnest", "heartfelt", "honest"],
    "polite": ["courteous", "respectful", "well-mannered", "formal"],
    "traditional": ["gallant", "old-fashioned", "chivalrous"],
    "physical": ["touchy", "sensual", "tactile", "passionate"]
}

# Reverse lookup for synonyms
SYNONYM_LOOKUP = {syn: key for key, syns in TRAIT_SYNONYMS.items() for syn in syns}
for canonical in TRAIT_SYNONYMS:
    SYNONYM_LOOKUP[canonical] = canonical  # self-mapping

# === FUNCTIONS ===
def normalize_trait(trait: str) -> str:
    """Lowercase, strip, remove punctuation, map synonyms."""
    t = trait.strip().lower()
    t = re.sub(r"[^\w\s]", "", t)  # remove punctuation
    return SYNONYM_LOOKUP.get(t, t)

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

# === LOAD DATASET ===
print("[INFO] Loading dataset...")
dataset = load_dataset(DATASET_NAME)
dataset = dataset["train"].filter(lambda x: x.get("category") == CATEGORY_FILTER)

# Assign IDs
for idx, row in enumerate(dataset):
    row["id"] = idx

# === LOAD MISTRAL ===
print("[INFO] Loading Mistral model...")
device = detect_system_device()
mistral = AutoModelForCausalLM.from_pretrained(
    Path(MODEL_PATH).parent,
    model_file=Path(MODEL_PATH).name,
    model_type="mistral",
    gpu_layers=50 if device != "cpu" else 0
)

# === EXTRACT TRAITS ===
print("[INFO] Extracting traits from conversations...")
traits_per_conv = {}
for row in tqdm(dataset, desc="Extracting traits"):
    prompt = (
        "Extract personality or flirting traits from the following text.\n"
        "Return a comma-separated list of short trait words.\n\n"
        f"Conversation:\n{row['text']}\n"
        "Traits:"
    )
    output = mistral(prompt, max_new_tokens=64, temperature=0.0).strip()
    traits = [normalize_trait(t) for t in output.split(",") if t.strip()]
    traits_per_conv[row["id"]] = traits

# === COUNT TRAITS ===
print("[INFO] Counting trait frequencies...")
all_traits = [t for traits in traits_per_conv.values() for t in traits]
trait_counts = Counter(all_traits)

# === FILTER BY MIN_FREQ ===
valid_traits = {t for t, c in trait_counts.items() if c >= MIN_FREQ}

# === EMBEDDING MODEL ===
print("[INFO] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Prepare training data (manual mapping first)
print("[INFO] Preparing embedding training data...")
train_texts = []
train_labels = []
for trait, style in SYNONYM_LOOKUP.items():
    if style in STYLES:
        train_texts.append(trait)
        train_labels.append(style)

X_train = embedder.encode(train_texts, convert_to_numpy=True)
y_train = train_labels

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# === CLASSIFY TRAITS ===
print("[INFO] Classifying traits via embedding + RandomForest...")
trait_to_style = {}
for trait in tqdm(valid_traits, desc="Classifying traits"):
    vec = embedder.encode([trait], convert_to_numpy=True)
    style = clf.predict(vec)[0]
    trait_to_style[trait] = style

# === PROPAGATE STYLES TO CONVERSATIONS ===
print("[INFO] Propagating styles to conversations...")
conv_styles = defaultdict(list)
for conv_id, traits in traits_per_conv.items():
    styles = {trait_to_style[t] for t in traits if t in trait_to_style}
    conv_styles[conv_id] = list(styles)

# === COUNT CONVERSATIONS PER STYLE ===
style_counts = Counter()
for styles in conv_styles.values():
    for s in styles:
        style_counts[s] += 1

print("[RESULT] Conversation counts per style:")
total_convs = len(conv_styles)
for style, count in style_counts.items():
    print(f"{style:<12}: {count}")

# === LOOSE RECLASSIFICATION WITH MISTRAL (Option 1) ===
print("\n[INFO] Starting loose reclassification for low-count styles...")

# Threshold for "low" representation
threshold = total_convs * 0.15
low_styles = {s for s, c in style_counts.items() if c < threshold}
print(f"[INFO] Low representation styles: {low_styles}")

# Find traits currently mapped to low styles OR not mapped at all
low_or_unclassified = [
    t for t in valid_traits
    if t not in trait_to_style or trait_to_style[t] in low_styles
]

for trait in tqdm(low_or_unclassified, desc="Loose classification"):
    prompt = (
        "Classify the following personality or flirting trait into one of these styles:\n"
        f"{', '.join(low_styles)}\n\n"
        f"Trait: {trait}\n\n"
        "Respond with only one style name from the list."
    )
    output = mistral(prompt, max_new_tokens=16, temperature=0.7).strip()
    if output in low_styles:
        trait_to_style[trait] = output

# Propagate new loose classifications
for conv_id, traits in traits_per_conv.items():
    for t in traits:
        if t in trait_to_style and trait_to_style[t] not in conv_styles[conv_id]:
            conv_styles[conv_id].append(trait_to_style[t])

# Final count
final_counts = Counter()
for styles in conv_styles.values():
    for s in styles:
        final_counts[s] += 1

print("\n[RESULT] Final conversation counts per style (after loose pass):")
for style, count in final_counts.items():
    print(f"{style:<12}: {count}")

# Save final results
with open("classified_conversations_final.json", "w", encoding="utf-8") as f:
    json.dump(conv_styles, f, indent=2, ensure_ascii=False)

print("[DONE] Pipeline with loose reclassification completed.")
