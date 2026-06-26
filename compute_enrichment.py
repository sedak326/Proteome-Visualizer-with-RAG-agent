"""
Compute per-cluster species enrichment (hypergeometric test) and save to
cluster_enrichment.csv. Run this once after recluster_kmeans.py.
"""
import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

# Map CSV species label → (app display name, hex color)
SPECIES_MAP = {
    'Bottlenose': ('Bottlenose Dolphin', '#4363D8'),
    'Graywhale':  ('Gray Whale',         '#3CB44B'),
    'Orca':       ('Orca',               '#F032E6'),
    'Harborseal': ('Harbor Seal',        '#9A6324'),
    'Sealion':    ('Sea Lion',           '#E6194B'),
    'Polarbear':  ('Polar Bear',         '#46F0F0'),
    'Hippo':      ('Hippopotamus',       '#F58231'),
}

UMAP_FILES = {
    'Bottlenose': 'umap_output/Bottlenose Dolphin_umap.csv',
    'Graywhale':  'umap_output/Graywhale_umap.csv',
    'Orca':       'umap_output/Killer whale_umap.csv',
    'Harborseal': 'umap_output/Harborseal_umap.csv',
    'Sealion':    'umap_output/California Sealion_umap.csv',
    'Polarbear':  'umap_output/Polar Bear_umap.csv',
    'Hippo':      'umap_output/Hippopotamus_umap.csv',
}

all_dfs = {sp: pd.read_csv(p) for sp, p in UMAP_FILES.items()}
combined = pd.concat(
    [df.assign(species=sp) for sp, df in all_dfs.items()],
    ignore_index=True
)

N = len(combined)
species_totals = combined['species'].value_counts().to_dict()
clusters = sorted(combined['Cluster Label'].unique())
cluster_names = (combined.drop_duplicates('Cluster Label')
                         .set_index('Cluster Label')['Cluster Name']
                         .to_dict())

print(f"Running enrichment: {len(clusters)} clusters × {len(species_totals)} species")

# Hypergeometric test for each species × cluster
rows = []
for sp, sp_total in species_totals.items():
    for lbl in clusters:
        mask = combined['Cluster Label'] == lbl
        M = mask.sum()
        k = (mask & (combined['species'] == sp)).sum()
        pval = hypergeom.sf(k - 1, N, sp_total, M)
        rows.append({'species': sp, 'cluster': lbl, 'k': k, 'M': M, 'pval': pval})

df_tests = pd.DataFrame(rows)

# BH correction within each species
corrected = []
for sp in species_totals:
    sub = df_tests[df_tests['species'] == sp].copy().reset_index(drop=True)
    _, padj, _, _ = multipletests(sub['pval'], method='fdr_bh')
    sub['padj'] = padj
    corrected.append(sub)
df_tests = pd.concat(corrected, ignore_index=True)

# For each cluster: collect all enriched species (padj < 0.05) and pick dominant
result_rows = []
for lbl in clusters:
    sub = df_tests[df_tests['cluster'] == lbl].copy()
    enriched = sub[sub['padj'] < 0.05].sort_values('padj')

    all_enriched = enriched['species'].tolist()
    all_enriched_display = [SPECIES_MAP[s][0] for s in all_enriched]
    all_enriched_colors = [SPECIES_MAP[s][1] for s in all_enriched]

    if enriched.empty:
        dominant_sp = ''
        dominant_display = ''
        dominant_color = '#666677'
        best_padj = 1.0
    else:
        dominant_sp = enriched.iloc[0]['species']
        dominant_display, dominant_color = SPECIES_MAP[dominant_sp]
        best_padj = enriched.iloc[0]['padj']

    result_rows.append({
        'Cluster Label':    lbl,
        'Cluster Name':     cluster_names.get(lbl, f'Cluster {lbl}'),
        'Dominant Species': dominant_display,
        'Dominant Color':   dominant_color,
        'All Enriched':     ', '.join(all_enriched_display),
        'N Enriched':       len(all_enriched),
        'Best padj':        best_padj,
    })

df_out = pd.DataFrame(result_rows)
df_out.to_csv('umap_output/cluster_enrichment.csv', index=False)

enriched_count = (df_out['Dominant Species'] != '').sum()
print(f"Enriched clusters: {enriched_count} / {len(clusters)}")
print(df_out[df_out['Dominant Species'] != ''].to_string(index=False))
