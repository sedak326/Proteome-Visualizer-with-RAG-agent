"""
Compute per-cluster silhouette scores on the 2D UMAP coordinates.

Uses a centroid-based approximation (O(n*k)) rather than exact pairwise
silhouette (O(n^2)) — feasible for 130k proteins × 370 clusters.

  a(i) = distance from protein i to its own cluster centroid
  b(i) = distance from protein i to the nearest OTHER cluster centroid
  s(i) = (b(i) - a(i)) / max(a(i), b(i))
  cluster score = mean s(i) over all proteins in the cluster

Saves umap_output/cluster_quality.csv with columns:
  Cluster Label, Cluster Name, N, Centroid X, Centroid Y,
  Mean Silhouette, Well Separated
"""
import numpy as np
import pandas as pd

UMAP_FILES = {
    'Sea Lion':    'umap_output/California Sealion_umap.csv',
    'Bottlenose':  'umap_output/Bottlenose Dolphin_umap.csv',
    'Gray Whale':  'umap_output/Graywhale_umap.csv',
    'Orca':        'umap_output/Killer whale_umap.csv',
    'Harbor Seal': 'umap_output/Harborseal_umap.csv',
    'Polar Bear':  'umap_output/Polar Bear_umap.csv',
    'Hippo':       'umap_output/Hippopotamus_umap.csv',
}

# Paper threshold for "well-separated"
SILHOUETTE_THRESHOLD = 0.5   # start moderate; Javi uses 0.95 for strict well-sep

print("Loading UMAP data...")
dfs = []
for sp, path in UMAP_FILES.items():
    df = pd.read_csv(path)
    df['species'] = sp
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
print(f"  {len(combined)} proteins, {combined['Cluster Label'].nunique()} clusters")

xy     = combined[['UMAP 1', 'UMAP 2']].values
labels = combined['Cluster Label'].values
cluster_names = (combined.drop_duplicates('Cluster Label')
                          .set_index('Cluster Label')['Cluster Name']
                          .to_dict())

unique_labels = np.unique(labels)
k = len(unique_labels)

# Build centroid matrix (k × 2)
print("Computing centroids...")
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
centroids = np.zeros((k, 2))
counts    = np.zeros(k, dtype=int)
for lbl, (x, y) in zip(labels, xy):
    i = label_to_idx[lbl]
    centroids[i] += [x, y]
    counts[i] += 1
centroids /= counts[:, None]

# For each protein: a(i) and b(i)
print("Computing centroid-based silhouette scores...")
protein_idx = np.array([label_to_idx[lbl] for lbl in labels])

# a(i): distance to own centroid
own_centroids = centroids[protein_idx]          # n × 2
a = np.linalg.norm(xy - own_centroids, axis=1) # n

# b(i): distance to nearest OTHER centroid
# Compute all pairwise centroid distances (k × k), then for each protein
# find the min distance to a centroid that isn't its own.
centroid_dists = np.linalg.norm(
    centroids[:, None, :] - centroids[None, :, :], axis=2
)  # k × k
np.fill_diagonal(centroid_dists, np.inf)
nearest_other_dist = centroid_dists.min(axis=1)  # k — nearest other centroid per cluster
b = nearest_other_dist[protein_idx]               # n

# Silhouette per protein
denom = np.maximum(a, b)
denom[denom == 0] = 1e-12
s = (b - a) / denom

# Aggregate per cluster
print("Aggregating per cluster...")
rows = []
for lbl in unique_labels:
    mask      = labels == lbl
    s_cluster = s[mask]
    rows.append({
        'Cluster Label':   lbl,
        'Cluster Name':    cluster_names.get(lbl, f'Cluster {lbl}'),
        'N':               int(mask.sum()),
        'Centroid X':      round(float(centroids[label_to_idx[lbl], 0]), 4),
        'Centroid Y':      round(float(centroids[label_to_idx[lbl], 1]), 4),
        'Mean Silhouette': round(float(s_cluster.mean()), 4),
        'Well Separated':  bool(s_cluster.mean() >= SILHOUETTE_THRESHOLD),
    })

df_out = pd.DataFrame(rows).sort_values('Mean Silhouette', ascending=False)
df_out.to_csv('umap_output/cluster_quality.csv', index=False)

n_well = df_out['Well Separated'].sum()
print(f"\nWell-separated clusters (S >= {SILHOUETTE_THRESHOLD}): {n_well} / {k}  "
      f"({100*n_well/k:.0f}%)")
print(f"\nTop 15 by silhouette score:")
print(df_out.head(15)[['Cluster Label','Cluster Name','N','Mean Silhouette']].to_string(index=False))
print(f"\nBottom 10 (fuzziest):")
print(df_out.tail(10)[['Cluster Label','Cluster Name','N','Mean Silhouette']].to_string(index=False))
print("\nSaved to umap_output/cluster_quality.csv")
