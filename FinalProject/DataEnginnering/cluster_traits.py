import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score

# ---- 1. Charger les traits extraits ----
with open("extracted_traits_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten + unique
all_traits = {trait for item in data for trait in item["extracted_traits"]}
traits_list = sorted(all_traits)

print(f"Chargé {len(traits_list)} traits depuis extracted_traits_cleaned.json")

# ---- 2. Génération des embeddings ----
print(f"\nGénération des embeddings...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(traits_list, show_progress_bar=True, convert_to_numpy=True)

# ---- 3. Paramètres à tester ----
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

# ---- 4. Recherche des meilleurs paramètres ----
results = []

for u_param in umap_params:
    print(f"UMAP params: {u_param}")
    umap_reducer = umap.UMAP(
        n_neighbors=u_param['n_neighbors'],
        min_dist=u_param['min_dist'],
        metric='cosine',
        random_state=42
    )
    embeddings_umap = umap_reducer.fit_transform(embeddings)

    for h_param in hdbscan_params:
        print(f"HDBSCAN params: {h_param}")
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

# ---- 5. Résultats triés ----
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by='silhouette_score', ascending=False)
print("\n=== Résultats triés par silhouette_score ===")
print(df_results)

# ---- 6. Meilleur modèle ----
best_row = df_results.iloc[0]
print(f"\nMeilleur score silhouette : {best_row['silhouette_score']:.4f} "
      f"avec UMAP n_neighbors={best_row['umap_n_neighbors']}, "
      f"min_dist={best_row['umap_min_dist']} et "
      f"HDBSCAN min_cluster_size={best_row['hdb_min_cluster_size']}, "
      f"min_samples={best_row['hdb_min_samples']}")

# ---- 7. Clustering final ----
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

# ---- 8. Organisation des clusters ----
clusters = defaultdict(list)
for trait, label in zip(traits_list, best_labels):
    clusters[label].append(trait)

print("\n=== Clusters du meilleur modèle ===")
for label, trait_list in sorted(clusters.items(), key=lambda x: (x[0] if x[0] != -1 else 9999)):
    if label == -1:
        print(f"Noise (-1): {len(trait_list)} traits")
    else:
        print(f"Cluster {label}: {len(trait_list)} traits")
    print(", ".join(trait_list[:30]))  # max 30 traits affichés

# ---- 9. Histogramme des tailles de clusters ----
cluster_sizes = Counter(best_labels)
plt.figure(figsize=(8,5))
plt.bar([str(k) for k in cluster_sizes.keys()], cluster_sizes.values())
plt.title("Taille des clusters")
plt.xlabel("Cluster ID")
plt.ylabel("Nombre de traits")
plt.tight_layout()
plt.show()

# ---- 10. Visualisation 2D ----
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
