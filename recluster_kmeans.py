"""
Re-cluster existing 2D UMAP coordinates with K-means.
K is chosen by maximizing silhouette score over a search range.
Updates the Cluster Label column in each per-species UMAP CSV and
adds a Cluster Name column derived from bi-gram analysis of protein names.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

UMAP_FILES = {
    'Bottlenose Dolphin': 'umap_output/Bottlenose Dolphin_umap.csv',
    'California Sealion': 'umap_output/California Sealion_umap.csv',
    'Graywhale':          'umap_output/Graywhale_umap.csv',
    'Harborseal':         'umap_output/Harborseal_umap.csv',
    'Killer whale':       'umap_output/Killer whale_umap.csv',
    'Polar Bear':         'umap_output/Polar Bear_umap.csv',
    'Hippopotamus':       'umap_output/Hippopotamus_umap.csv',
}

# K search range — silhouette score picks the best.
# After fixing UMAP parameters (n_neighbors=50, min_dist=0.0), the silhouette
# should peak somewhere in this range rather than monotonically increasing.
K_MIN = 20
K_MAX = 400
K_STEP = 10

# Silhouette is expensive on 130k points; subsample for scoring only
SILHOUETTE_SAMPLE = 10000
RANDOM_STATE = 42

# Words to skip when generating cluster labels from protein names
_STOP_WORDS = {
    'protein', 'domain', 'family', 'like', 'related', 'putative',
    'predicted', 'probable', 'homolog', 'isoform', 'subunit',
    'chain', 'precursor', 'fragment', 'uncharacterized', 'type',
    'and', 'the', 'of', 'with', 'to', 'a', 'an',
}


def find_best_k(coords: np.ndarray) -> tuple[int, float]:
    """Try k values from K_MIN to K_MAX, return (best_k, best_score)."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(coords), size=min(SILHOUETTE_SAMPLE, len(coords)), replace=False)
    sample = coords[idx]

    best_k, best_score = -1, -2.0
    print(f"  Searching k in [{K_MIN}, {K_MAX}] step {K_STEP}...")
    for k in range(K_MIN, K_MAX + 1, K_STEP):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=3, batch_size=4096)
        labels_sample = km.fit_predict(sample)
        score = silhouette_score(sample, labels_sample, sample_size=min(5000, len(sample)))
        print(f"    k={k:4d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"  -> Best k={best_k}  silhouette={best_score:.4f}")
    return best_k, best_score


def _tokenize(name: str) -> list[str]:
    """Split a protein name into lowercase words, dropping stop words and short tokens."""
    tokens = re.findall(r'[a-zA-Z]{3,}', name.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def build_cluster_names(labels: np.ndarray, protein_names: pd.Series) -> dict[int, str]:
    """For each cluster, pick the top bi-gram (or unigram fallback) from protein names."""
    cluster_names = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        names_in_cluster = protein_names[mask].dropna()
        if names_in_cluster.empty:
            cluster_names[lbl] = f'Cluster {lbl}'
            continue

        # Count unigrams and bigrams
        unigrams: Counter = Counter()
        bigrams: Counter = Counter()
        for name in names_in_cluster:
            tokens = _tokenize(str(name))
            unigrams.update(tokens)
            bigrams.update(zip(tokens, tokens[1:]))

        if bigrams:
            top_bigram = bigrams.most_common(1)[0][0]
            label = ' '.join(top_bigram).title()
        elif unigrams:
            label = unigrams.most_common(1)[0][0].title()
        else:
            label = f'Cluster {lbl}'

        cluster_names[lbl] = label

    return cluster_names


def main():
    # Load all species and stack 2D coords
    print("Loading all UMAP CSVs...")
    all_dfs = {}
    for name, path in UMAP_FILES.items():
        df = pd.read_csv(path)
        all_dfs[name] = df
        print(f"  {name}: {len(df)} proteins")

    combined_coords = np.vstack([df[['UMAP 1', 'UMAP 2']].values for df in all_dfs.values()])
    combined_names = pd.concat([df['Protein names'] for df in all_dfs.values()], ignore_index=True)
    print(f"\nCombined: {len(combined_coords)} proteins total")

    # Find best k on combined coords
    print("\nFinding optimal k via silhouette score...")
    best_k, best_score = find_best_k(combined_coords)

    # Fit final K-means on full dataset with best k
    print(f"\nFitting K-means with k={best_k} on all {len(combined_coords)} proteins...")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=5, max_iter=500)
    all_labels = km.fit_predict(combined_coords)
    print("Done.")

    # Build cluster names from bi-gram analysis
    print("\nBuilding cluster names from protein name bi-grams...")
    cluster_name_map = build_cluster_names(all_labels, combined_names)

    # Check species composition of clusters
    species_list = []
    for name, df in all_dfs.items():
        species_list.extend([name] * len(df))
    species_arr = np.array(species_list)

    unique_labels = np.unique(all_labels)
    species_exclusive = sum(
        1 for lbl in unique_labels
        if len(set(species_arr[all_labels == lbl])) == 1
    )
    all_species = sum(
        1 for lbl in unique_labels
        if len(set(species_arr[all_labels == lbl])) == 7
    )

    print(f"\nCluster composition (k={best_k}):")
    print(f"  Total clusters: {len(unique_labels)}")
    print(f"  Clusters with only 1 species: {species_exclusive} ({100*species_exclusive/len(unique_labels):.1f}%)")
    print(f"  Clusters with all 7 species:  {all_species} ({100*all_species/len(unique_labels):.1f}%)")

    print("\nSample cluster names:")
    for lbl in list(unique_labels)[:10]:
        mask = all_labels == lbl
        n_species = len(set(species_arr[mask]))
        print(f"  Cluster {lbl:4d} ({mask.sum():5d} proteins, {n_species} species): {cluster_name_map[lbl]}")

    # Write updated CSVs
    print("\nWriting updated CSVs...")
    offset = 0
    for name, df in all_dfs.items():
        n = len(df)
        df = df.copy()
        df['Cluster Label'] = all_labels[offset:offset + n]
        df['Cluster Name'] = [cluster_name_map[lbl] for lbl in all_labels[offset:offset + n]]
        out_path = UMAP_FILES[name]
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}")
        offset += n

    print(f"\nDone. All CSVs updated with k={best_k} K-means labels and bi-gram cluster names.")


if __name__ == '__main__':
    main()
