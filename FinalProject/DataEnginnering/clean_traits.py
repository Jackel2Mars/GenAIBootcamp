import json

INPUT_JSON = "extracted_traits.json"
OUTPUT_JSON = "extracted_traits_cleaned.json"

# Dictionnaire de remplacements
replace_map = {
    "self_love": "self-love",
    "believes in happily ever afters": "romantic",
    "believes-in-true-love": "romantic",
    "encourages dreams and aspirations": "nurturing",
    "inspires creativity and passion": "inspiring"  # ou nurturing si tu préfères
}

# Traits à supprimer
remove_set = {"sweet-toothed"}

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    traits = entry.get("extracted_traits", [])
    cleaned_traits = []
    for t in traits:
        t_stripped = t.strip()
        if t_stripped in remove_set:
            continue
        if t_stripped in replace_map:
            cleaned_traits.append(replace_map[t_stripped])
        else:
            cleaned_traits.append(t_stripped)
    entry["extracted_traits"] = cleaned_traits

# Sauvegarde
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[OK] Fichier nettoyé sauvegardé dans {OUTPUT_JSON}")
