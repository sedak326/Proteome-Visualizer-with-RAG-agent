"""
Compute group-level enrichment (Pinniped / Cetacean / Hippo / Carnivore)
using hypergeometric test + BH correction.

Saves umap_output/group_enrichment.csv
"""
import pandas as pd
import numpy as np
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

GROUP_MAP = {
    'Sea Lion':    'Pinniped',
    'Harbor Seal': 'Pinniped',
    'Bottlenose':  'Cetacean',
    'Gray Whale':  'Cetacean',
    'Orca':        'Cetacean',
    'Polar Bear':  'Carnivore',
    'Hippo':       'Hippo',
}

GROUP_COLORS = {
    'Pinniped':  '#C77DFF',  # violet
    'Cetacean':  '#4CC9F0',  # sky blue
    'Hippo':     '#FF9F1C',  # amber
    'Carnivore': '#06D6A0',  # mint green
}

print("Loading data...")
dfs = []
for sp, path in UMAP_FILES.items():
    df = pd.read_csv(path)[['Entry', 'Cluster Label', 'Cluster Name']]
    df['species'] = sp
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
combined['group'] = combined['species'].map(GROUP_MAP)

N = len(combined)
groups = ['Pinniped', 'Cetacean', 'Hippo', 'Carnivore']
group_totals = {g: (combined['group'] == g).sum() for g in groups}
clusters = sorted(combined['Cluster Label'].unique())
cluster_names = (combined.drop_duplicates('Cluster Label')
                          .set_index('Cluster Label')['Cluster Name'].to_dict())

print(f"  {N} proteins, {len(clusters)} clusters, {len(groups)} groups")

# Hypergeometric test
rows = []
for grp in groups:
    K = group_totals[grp]
    for cl in clusters:
        mask_cl = combined['Cluster Label'] == cl
        M = mask_cl.sum()
        k = (mask_cl & (combined['group'] == grp)).sum()
        pval = hypergeom.sf(k - 1, N, K, M)
        rows.append({'group': grp, 'cluster': cl, 'k': k, 'M': M, 'pval': pval})

df_tests = pd.DataFrame(rows)

# BH correction within each group
corrected = []
for grp in groups:
    sub = df_tests[df_tests['group'] == grp].copy().reset_index(drop=True)
    _, padj, _, _ = multipletests(sub['pval'], method='fdr_bh')
    sub['padj'] = padj
    corrected.append(sub)
df_tests = pd.concat(corrected, ignore_index=True)

# Per cluster: dominant group (lowest padj < 0.05), all enriched groups
result_rows = []
for cl in clusters:
    sub = df_tests[df_tests['cluster'] == cl].copy()
    enriched = sub[sub['padj'] < 0.05].sort_values('padj')
    all_groups = enriched['group'].tolist()

    if enriched.empty:
        dominant_grp   = ''
        dominant_color = '#666677'
        best_padj      = 1.0
    else:
        dominant_grp   = enriched.iloc[0]['group']
        dominant_color = GROUP_COLORS[dominant_grp]
        best_padj      = enriched.iloc[0]['padj']

    # Per-group composition %
    total = combined[combined['Cluster Label'] == cl]
    n_total = len(total)
    comp = {g: round((total['group'] == g).sum() / n_total * 100, 1) for g in groups}

    result_rows.append({
        'Cluster Label':    cl,
        'Cluster Name':     cluster_names.get(cl, f'Cluster {cl}'),
        'Dominant Group':   dominant_grp,
        'Dominant Color':   dominant_color,
        'All Enriched':     ', '.join(all_groups),
        'N Enriched':       len(all_groups),
        'Best padj':        best_padj,
        'Pinniped %':       comp['Pinniped'],
        'Cetacean %':       comp['Cetacean'],
        'Hippo %':          comp['Hippo'],
        'Carnivore %':      comp['Carnivore'],
    })

df_out = pd.DataFrame(result_rows)
df_out.to_csv('umap_output/group_enrichment.csv', index=False)

print("\nEnriched clusters per group (padj < 0.05):")
for grp in groups:
    n = (df_out['Dominant Group'] == grp).sum()
    print(f"  {grp}: {n}")

for grp in groups:
    sub = df_out[df_out['Dominant Group'] == grp].sort_values('Best padj')
    if sub.empty:
        continue
    print(f"\n=== {grp} enriched ===")
    for _, r in sub.iterrows():
        print(f"  Cl {int(r['Cluster Label']):>3}  {r['Cluster Name']:<28}  "
              f"Pin={r['Pinniped %']}% Cet={r['Cetacean %']}% "
              f"Hip={r['Hippo %']}% Car={r['Carnivore %']}%  "
              f"padj={r['Best padj']:.2e}")

print("\nSaved to umap_output/group_enrichment.csv")
