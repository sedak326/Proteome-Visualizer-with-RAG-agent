"""
Compute per-cluster annotation enrichment from protein names.

For each meaningful keyword extracted from UniProt protein names, runs a
hypergeometric test (+ BH correction) across all K-means clusters. The most
enriched term per cluster becomes its functional annotation label.

Saves umap_output/annotation_enrichment.csv with columns:
  Cluster Label, Cluster Name, Top Term, Top padj, All Terms (semicolon-sep)
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

UMAP_FILES = {
    'Sea Lion':    'umap_output/California Sealion_umap.csv',
    'Bottlenose':  'umap_output/Bottlenose Dolphin_umap.csv',
    'Gray Whale':  'umap_output/Graywhale_umap.csv',
    'Orca':        'umap_output/Killer whale_umap.csv',
    'Harbor Seal': 'umap_output/Harborseal_umap.csv',
    'Polar Bear':  'umap_output/Polar Bear_umap.csv',
    'Hippo':       'umap_output/Hippopotamus_umap.csv',
}

MIN_TERM_COUNT = 100  # ignore terms rarer than this across all proteins
PADJ_THRESHOLD = 0.05

_STOP = {
    'protein', 'domain', 'family', 'like', 'related', 'putative', 'predicted',
    'probable', 'homolog', 'isoform', 'subunit', 'chain', 'precursor',
    'fragment', 'uncharacterized', 'type', 'and', 'the', 'of', 'with', 'to',
    'a', 'an', 'low', 'quality', 'repeat', 'containing', 'member', 'factor',
    'binding', 'associated', 'expressed', 'novel', 'hypothetical', 'gene',
    'superfamily', 'motif', 'region', 'very', 'long', 'short', 'small',
    'large', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'sigma',
    'transporting', 'similar', 'molecule', 'complex', 'activity',
    'dependent', 'specific', 'negative', 'positive',
}

def tokenize(name: str) -> list[str]:
    tokens = re.findall(r'[a-zA-Z]{4,}', name.lower())
    return [t for t in tokens if t not in _STOP]

print("Loading UMAP CSVs...")
dfs = []
for sp, path in UMAP_FILES.items():
    df = pd.read_csv(path, usecols=['Entry', 'Protein names', 'Cluster Label'])
    df['species'] = sp
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined = combined[combined['Cluster Label'] >= 0].copy()
print(f"  {len(combined)} proteins, {combined['Cluster Label'].nunique()} clusters")

# Load cluster names from one CSV (all species share the same cluster names)
_tmp = pd.read_csv(list(UMAP_FILES.values())[0], usecols=['Cluster Label', 'Cluster Name'])
cluster_name_map = _tmp.drop_duplicates('Cluster Label').set_index('Cluster Label')['Cluster Name'].to_dict()

# Extract keywords for each protein
print("Extracting keywords from protein names...")
protein_terms = []  # list of sets, one per protein row
for name in combined['Protein names'].fillna(''):
    protein_terms.append(set(tokenize(str(name))))

# Build global term counts
all_term_counts: Counter = Counter()
for terms in protein_terms:
    all_term_counts.update(terms)

vocab = [t for t, c in all_term_counts.items() if c >= MIN_TERM_COUNT]
print(f"  Vocabulary size (≥{MIN_TERM_COUNT} occurrences): {len(vocab)} terms")

# Build binary presence matrix: protein × term
# Use index arrays for speed
term_to_idx = {t: i for i, t in enumerate(vocab)}
V = len(vocab)
N = len(combined)
labels = combined['Cluster Label'].values
clusters = sorted(combined['Cluster Label'].unique())
K_clusters = len(clusters)

print(f"  Building presence arrays for {V} terms × {N} proteins...")
# For each term, store array of row indices where term is present
term_rows = {t: [] for t in vocab}
for row_i, terms in enumerate(protein_terms):
    for t in terms:
        if t in term_to_idx:
            term_rows[t].append(row_i)
term_presence = {t: np.array(rows) for t, rows in term_rows.items()}

# Global term counts
term_K = {t: len(idxs) for t, idxs in term_presence.items()}

print(f"Running hypergeometric tests ({V} terms × {K_clusters} clusters)...")
rows_out = []
for t in vocab:
    K_t = term_K[t]
    present_set = set(term_presence[t])
    for cl in clusters:
        mask = labels == cl
        cl_idx = np.where(mask)[0]
        M = len(cl_idx)
        k = len(set(cl_idx) & present_set)
        pval = hypergeom.sf(k - 1, N, K_t, M)
        expected = M * K_t / N
        fc = k / (expected + 1e-9)
        rows_out.append({'term': t, 'cluster': cl, 'k': k, 'M': M,
                         'expected': round(expected, 2), 'fc': fc, 'pval': pval})

df_tests = pd.DataFrame(rows_out)

print("Applying BH correction per term...")
corrected = []
for t in vocab:
    sub = df_tests[df_tests['term'] == t].copy().reset_index(drop=True)
    _, padj, _, _ = multipletests(sub['pval'], method='fdr_bh')
    sub['padj'] = padj
    corrected.append(sub)
df_tests = pd.concat(corrected, ignore_index=True)

print("Building per-cluster results...")
result_rows = []
for cl in clusters:
    sig = df_tests[(df_tests['cluster'] == cl) & (df_tests['padj'] < PADJ_THRESHOLD)]
    sig = sig.sort_values(['padj', 'fc'], ascending=[True, False])

    if sig.empty:
        top_term = ''
        top_padj = 1.0
        all_terms = ''
    else:
        top_term = sig.iloc[0]['term']
        top_padj = float(sig.iloc[0]['padj'])
        all_terms = ';'.join(sig['term'].tolist()[:10])

    result_rows.append({
        'Cluster Label': cl,
        'Cluster Name':  cluster_name_map.get(cl, f'Cluster {cl}'),
        'Top Term':      top_term,
        'Top padj':      top_padj,
        'All Terms':     all_terms,
    })

df_out = pd.DataFrame(result_rows)
df_out.to_csv('umap_output/annotation_enrichment.csv', index=False)

n_annotated = (df_out['Top Term'] != '').sum()
print(f"\nAnnotated clusters: {n_annotated} / {K_clusters} ({100*n_annotated/K_clusters:.0f}%)")
print("\nSample annotations:")
for _, r in df_out[df_out['Top Term'] != ''].head(15).iterrows():
    print(f"  Cl {int(r['Cluster Label']):>3}  {r['Cluster Name']:<28}  → {r['Top Term']}  (padj={r['Top padj']:.2e})")

print("\nSaved to umap_output/annotation_enrichment.csv")
