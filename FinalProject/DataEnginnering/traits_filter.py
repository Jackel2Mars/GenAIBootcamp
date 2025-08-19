from datasets import load_dataset
from collections import Counter
import spacy
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

# Tags à exclure
VERB_TAGS = {"VB", "VBD", "VBP", "VBZ"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}

def filter_romantic_traits(dataset_name):
    ds = load_dataset(dataset_name, split="train")

    # Étape 1 : Filtrer sur romance
    filtered = [
        entry for entry in ds
        if any("romance" in cat.lower() for cat in entry.get("categories", []))
    ]

    final_traits = []
    style_counts = Counter()

    for entry in filtered:
        personalities = entry.get("personalities", [])[:2]  # Étape 2 : garder 2 premiers
        for style in personalities:
            doc = nlp(style.lower())

            # Étape 3 : Rejeter si contient un verbe
            if any(token.tag_ in VERB_TAGS for token in doc):
                continue

            # Étape 4 : Rejeter si contient un nom (même en premier mot)
            if any(token.tag_ in NOUN_TAGS for token in doc):
                continue

            final_traits.append(style)
            style_counts[style.lower()] += 1

    return final_traits, style_counts

# Exemple sur un dataset
traits, counts = filter_romantic_traits("AlekseyKorshuk/synthetic-romantic-characters")

# Visualisation complète
if counts:
    styles, freqs = zip(*counts.most_common())
    plt.figure(figsize=(10, 6))
    plt.bar(styles, freqs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Occurrences")
    plt.title("Distribution des traits filtrés (romance)")
    plt.tight_layout()
    plt.show()

print(f"{len(traits)} traits retenus.")
