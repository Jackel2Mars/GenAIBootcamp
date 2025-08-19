import json
from collections import Counter, defaultdict
from datasets import load_dataset
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt
import pandas as pd

# Embedding & clustering libs
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np

from sklearn.metrics import silhouette_score

# ---- Ollama helper (streaming JSONL parser) ----
def parse_ollama_streaming_response(raw_text):
    full_response = ""
    for line in raw_text.strip().splitlines():
        try:
            obj = json.loads(line)
            if "response" in obj:
                full_response += obj["response"]
            if obj.get("done", False):
                break
        except json.JSONDecodeError:
            continue
    return full_response

import requests
import json

def extract_personality_traits_batch_http(batch_personalities, model_name="granite3.3:2b", max_tokens=100, temperature=0.2):
    """
    batch_personalities: list of lists (each inner list = personalities of one entry)
    returns: list of lists (extracted traits per entry)
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    results = []
    for personalities in batch_personalities:
        prompt = f"""
{{
  "messages": [
    {{"role": "control", "content": "text extraction"}},
    {{"role": "user", "content": "You are an expert psychologist specialized in extracting core behavioral personality traits from text descriptions of people.\n\nGiven a list of personality descriptions, extract only the core personality traits, i.e. character-related qualities or stable behavioral tendencies.\n\nDo NOT extract interests, hobbies, professions, physical descriptions, preferences, or subjective opinions.\n\nReturn your answer as a JSON list of simple lowercase strings, each string being a single trait.\n\nList:\n{json.dumps(personalities, ensure_ascii=False)}"}}
  ]
}}
"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
        except Exception as e:
            print(f"Request error: {e}")
            results.append([])
            continue

        if resp.status_code == 200:
            raw_text = resp.text.strip()
            completion = parse_ollama_streaming_response(raw_text)
            try:
                traits = json.loads(completion)
                # normalize: lowercase and strip
                traits = [t.strip().lower() for t in traits if isinstance(t, str) and t.strip()]
            except Exception as e:
                print(f"Erreur JSON parsing pour completion:\n{completion}\nErreur: {e}")
                traits = []
        else:
            print(f"Ollama error status {resp.status_code}: {resp.text[:200]}")
            traits = []

        results.append(traits)
    return results

# ---- 1. Charger dataset et ajouter ID colonne proprement ----
ds = load_dataset("AlekseyKorshuk/synthetic-romantic-characters", split="train")
ds = ds.map(lambda example, idx: {"id": f"conv_{idx}"}, with_indices=True)

# ---- 2. Filtrer romance ----
romance_entries = [entry for entry in ds if any("romance" in c.lower() for c in entry.get("categories", []))]

# ---- 3. Extraction traits (batch) ----
batch_size = 16
romance_traits = set()

print("Extraction des traits depuis les entrées 'romance' (LLM)...")
for i in tqdm(range(0, len(romance_entries), batch_size), desc="Extraction traits romance"):
    batch = romance_entries[i : i + batch_size]
    batch_personalities = [entry.get("personalities", []) for entry in batch]
    extracted = extract_personality_traits_batch_http(batch_personalities)
    for traits in extracted:
        # dédup par entrée avant update (évite répétitions inutiles)
        romance_traits.update(set(traits))

print(f"Nombre de traits d'intérêt (romance): {len(romance_traits)}")

# ---- 4. Comptage global sur tout le dataset ----
freq_total = Counter()
freq_convos = defaultdict(set)

print("Comptage global (tous contexts) des traits romance...")
for entry in tqdm(ds, desc="Comptage global"):
    convo_id = entry["id"]
    personalities = [p.lower() for p in entry.get("personalities", [])]
    # si la liste personalities est longue, on la transforme en set pour lookup rapide
    personalities_set = set(personalities)
    for trait in romance_traits:
        if trait in personalities_set:
            freq_total[trait] += 1
            freq_convos[trait].add(convo_id)

# Affichage rapide
print("\nTrait\tTotal Occurrences\t# Conversations")
for trait, total in freq_total.most_common(40):
    print(f"{trait}\t{total}\t{len(freq_convos[trait])}")

# Optionnel: inspect example convo ids for a given trait
example_trait = "adventurous"
print(f"\nExemples de convo ids pour '{example_trait}': {list(freq_convos[example_trait])[:10]} (total {len(freq_convos[example_trait])})")

# ---- 5. Embeddings (traits) ----
traits_list = sorted(list(romance_traits))
print(f"\nGénération des embeddings pour {len(traits_list)} traits...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(traits_list, show_progress_bar=True, convert_to_numpy=True)

# Exemples d'hyperparamètres à tester
umap_params = [
    {'n_neighbors': 5, 'min_dist': 0.1},
    {'n_neighbors': 10, 'min_dist': 0.1},
    {'n_neighbors': 15, 'min_dist': 0.1},
    {'n_neighbors': 10, 'min_dist': 0.5},
]

hdbscan_params = [
    {'min_cluster_size': 5, 'min_samples': 5},
    {'min_cluster_size': 10, 'min_samples': 10},
    {'min_cluster_size': 20, 'min_samples': 10},
    {'min_cluster_size': 30, 'min_samples': 15},
]

results = []

for u_param in umap_params:
    print(f"UMAP params: n_neighbors={u_param['n_neighbors']}, min_dist={u_param['min_dist']}")
    umap_reducer = umap.UMAP(
        n_neighbors=u_param['n_neighbors'],
        min_dist=u_param['min_dist'],
        metric='cosine',
        random_state=42
    )
    embeddings_umap = umap_reducer.fit_transform(embeddings)

    for h_param in hdbscan_params:
        print(f"HDBSCAN params: min_cluster_size={h_param['min_cluster_size']}, min_samples={h_param['min_samples']}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=h_param['min_cluster_size'],
            min_samples=h_param['min_samples'],
            metric='euclidean',
            prediction_data=False
        )
        labels = clusterer.fit_predict(embeddings_umap)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if n_clusters > 1:
            score = silhouette_score(embeddings_umap[labels != -1], labels[labels != -1])
        else:
            score = np.nan

        results.append({
            'umap_n_neighbors': u_param['n_neighbors'],
            'umap_min_dist': u_param['min_dist'],
            'hdb_min_cluster_size': h_param['min_cluster_size'],
            'hdb_min_samples': h_param['min_samples'],
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': score
        })

# Convertir en DataFrame et trier par silhouette_score décroissant
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='silhouette_score', ascending=False)
print("\n=== Résultats de recherche d'hyperparamètres triés par silhouette_score ===")
print(df_results)

# Optionnel : afficher les clusters du meilleur modèle
best_row = df_results.iloc[0]
print(f"\nMeilleur score silhouette : {best_row['silhouette_score']:.4f} avec UMAP n_neighbors={best_row['umap_n_neighbors']}, min_dist={best_row['umap_min_dist']} et HDBSCAN min_cluster_size={best_row['hdb_min_cluster_size']}, min_samples={best_row['hdb_min_samples']}")

# Refaire le clustering avec les meilleurs paramètres pour inspection / visualisation
best_umap = umap.UMAP(
    n_neighbors=int(best_row['umap_n_neighbors']),
    min_dist=best_row['umap_min_dist'],
    metric='cosine',
    random_state=42
)
embeddings_umap_best = best_umap.fit_transform(embeddings)

best_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size=int(best_row['hdb_min_cluster_size']),
    min_samples=int(best_row['hdb_min_samples']),
    metric='euclidean',
    prediction_data=False
)
best_labels = best_hdbscan.fit_predict(embeddings_umap_best)

# Affichage clusters du meilleur modèle
clusters = defaultdict(list)
for trait, label in zip(traits_list, best_labels):
    clusters[label].append(trait)

print("\n=== Clusters du meilleur modèle ===")
for label, trait_list in sorted(clusters.items(), key=lambda x: (x[0] if x[0] != -1 else 9999)):
    if label == -1:
        print(f"Noise (-1): {len(trait_list)} traits")
    else:
        print(f"Cluster {label}: {len(trait_list)} traits")

for label, trait_list in clusters.items():
    print("\n---")
    if label == -1:
        print("Noise (unclustered):")
    else:
        print(f"Cluster {label}:")
    print(", ".join(trait_list[:30]))  # Affiche max 30 traits par cluster

# Visualisation 2D
plt.figure(figsize=(10, 7))
unique_labels = np.unique(best_labels)
palette = plt.cm.get_cmap("tab10", len(unique_labels))
for i, label in enumerate(unique_labels):
    mask = best_labels == label
    color = 'lightgray' if label == -1 else palette(i)
    label_name = "Noise" if label == -1 else f"Cluster {label}"
    plt.scatter(embeddings_umap_best[mask, 0], embeddings_umap_best[mask, 1], color=color, label=label_name, s=60, alpha=0.8)
for xi, yi, t in zip(embeddings_umap_best[:,0], embeddings_umap_best[:,1], traits_list):
    plt.text(xi, yi, t, fontsize=8, alpha=0.85)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("UMAP + HDBSCAN Clustering - Meilleur modèle")
plt.tight_layout()
plt.show()