import json
from pathlib import Path
from collections import defaultdict

STYLES = ["Traditional", "Playful", "Physical", "Sincere", "Polite", "None"]
OUTPUT_DIR = Path("styles_traits_forced")
ANNOTATED_FILE = Path("annotated_traits_forced.json")

def load_json_safe(p: Path):
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"[warn] Impossible de parser {p}: {e}")
            return None

def count_items(items):
    """items = liste d'objets {'trait': {...}, 'style': 'Playful'}"""
    if not isinstance(items, list):
        return 0, 0
    ids = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        trait = it.get("trait", {})
        conv_id = trait.get("id")
        if conv_id is not None:
            ids.add(conv_id)
    return len(items), len(ids)

def ensure_per_style_files_from_annotated(annotated):
    """Reconstruit styles_datasets/*.json à partir d'annotated_styles.json si besoin."""
    if not isinstance(annotated, list):
        return
    OUTPUT_DIR.mkdir(exist_ok=True)
    by_style = defaultdict(list)
    for it in annotated:
        s = it.get("style")
        if s in STYLES:
            by_style[s].append(it)
    for s in STYLES:
        out = OUTPUT_DIR / f"{s.lower()}.json"
        if not out.exists() or not load_json_safe(out):
            with out.open("w", encoding="utf-8") as f:
                json.dump(by_style[s], f, indent=2, ensure_ascii=False)
            print(f"[io] Reconstruit {out} avec {len(by_style[s])} items.")

# 1) Charger le fichier annoté (s’il existe)
annotated = load_json_safe(ANNOTATED_FILE)

# 2) Si possible, reconstruire les fichiers par style à partir d'annotated
if annotated:
    ensure_per_style_files_from_annotated(annotated)

# 3) Comptage à partir d'annotated_styles.json (si dispo)
print("=== Résumé depuis annotated_styles.json ===")
if annotated:
    for s in STYLES:
        items_s = [x for x in annotated if isinstance(x, dict) and x.get("style") == s]
        n_items, n_conv = count_items(items_s)
        print(f"{s:<12} : {n_items:>5} items | {n_conv:>5} conversations uniques")
else:
    print("[info] annotated_styles.json introuvable ou vide.")

# 4) Comptage à partir des fichiers par style (si dispo)
print("\n=== Résumé depuis styles_datasets/*.json ===")
for s in STYLES:
    p = OUTPUT_DIR / f"{s.lower()}.json"
    data = load_json_safe(p)
    if data is None:
        print(f"{s:<12} : fichier manquant ({p})")
        continue
    n_items, n_conv = count_items(data)
    print(f"{s:<12} : {n_items:>5} items | {n_conv:>5} conversations uniques (source: {p.name})")

# 5) Petit check de cohérence
if annotated:
    total_ann = sum(count_items([x])[0] for x in annotated)  # = len(annotated)
    total_dir = 0
    for s in STYLES:
        p = OUTPUT_DIR / f"{s.lower()}.json"
        d = load_json_safe(p)
        if isinstance(d, list):
            total_dir += len(d)
    if total_dir != len(annotated):
        print(f"\n[warn] Somme des items par style ({total_dir}) ≠ items annotés ({len(annotated)}).")
    else:
        print("\n[ok] Somme des items par style = items annotés.")
