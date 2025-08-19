import json
import re

# Fichier en entrée
INPUT_JSON = "extracted_traits_cleaned.json"

# Lecture du JSON
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraction et mise à plat de tous les traits
all_traits = set()
for entry in data:
    traits = entry.get("extracted_traits", [])
    for trait in traits:
        all_traits.add(trait.strip())

# Filtre : trop long ou contient un caractère non alphabétique
non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")  # autorise lettres + espaces

problematic_traits = sorted([
    t for t in all_traits
    if len(t) > 30 or non_alpha_pattern.search(t)
])

# Affichage
print(f"Nombre de traits problématiques : {len(problematic_traits)}")
for trait in problematic_traits:
    print(trait)
