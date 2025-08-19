import spacy
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

# Charger le modèle spaCy en anglais
nlp = spacy.load("en_core_web_sm")

# Charger le dataset complet
ds = load_dataset("AlekseyKorshuk/synthetic-romantic-characters", split="train")

# Filtrer les entrées où "romance" apparaît dans les catégories
filtered = [entry for entry in ds if any("romance" in cat.lower() for cat in entry.get("categories", []))]

styles_set = set()

def keep_style(style):
    doc = nlp(style)
    if not doc:
        return False
    first_token = doc[0]
    # On rejette si premier mot est VB, VBD, VBP ou VBZ
    if first_token.tag_ in ("VB", "VBD", "VBP", "VBZ"):
        return False
    return True

if not filtered:
    print("Aucune conversation filtrée avec catégorie 'romance'. Vérifier les noms de partout.")
else:
    style_counts = Counter()
    for entry in filtered:
        for style in entry.get("personalities", []):
            style_lower = style.lower()
            if keep_style(style_lower):
                styles_set.add(style_lower)
                style_counts[style_lower] += 1

    print("Styles conservés :", styles_set)

    # Visualiser la distribution sans limite de top N
    styles, counts = zip(*style_counts.most_common())
    plt.figure(figsize=(12, 8))
    plt.bar(styles, counts)
    plt.xticks(rotation=90, ha="right")
    plt.ylabel("Nombre d'occurrences dans les catégories 'romance'")
    plt.title("Distribution des styles de flirt filtrés")
    plt.tight_layout()
    plt.show()
