"""
Compute per-cluster OrthoDB ortholog-group enrichment (evolutionary/homology enrichment).

For each K-means cluster, runs a hypergeometric test (+ BH correction) against every
OrthoDB (v12, Mammalia node 40674) ortholog group present in the dataset, to check
whether PLVis clusters align with curated evolutionary/ortholog groupings -- the same
kind of test the original PLVis paper ran against OrthoDB and CATH FunFams.

Requires results/orthodb/<species>.og.annotations, produced by ODB-mapper
(see /uss/skavlak/plvis_homology_enrichment/run_orthodb_all.sh). Each file has
columns: #query, ODB_OG, evalue, score, COG_category, Description, GOs_mf, GOs_bp,
EC, KEGG_ko, Interpro -- one row per protein that mapped to an ortholog group.

Saves umap_output/orthodb_enrichment.csv with columns:
  Cluster Label, Cluster Name, Top OG, Top OG Description, Top padj, N Enriched OGs, All OGs (semicolon-sep)
"""
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

ORTHODB_FILES = {
    'Sea Lion':    '/uss/skavlak/plvis_homology_enrichment/results/orthodb/sealion.og.annotations',
    'Bottlenose':  '/uss/skavlak/plvis_homology_enrichment/results/orthodb/bottlenose.og.annotations',
    'Gray Whale':  '/uss/skavlak/plvis_homology_enrichment/results/orthodb/graywhale.og.annotations',
    'Orca':        '/uss/skavlak/plvis_homology_enrichment/results/orthodb/orca.og.annotations',
    'Harbor Seal': '/uss/skavlak/plvis_homology_enrichment/results/orthodb/harborseal.og.annotations',
    'Polar Bear':  '/uss/skavlak/plvis_homology_enrichment/results/orthodb/polarbear.og.annotations',
    'Hippo':       '/uss/skavlak/plvis_homology_enrichment/results/orthodb/hippo.og.annotations',
}

# Most OrthoDB Mammalia-level groups are near single-copy (at most one hit per
# species), so a group observed only once or twice across the combined dataset
# can never show meaningful cluster enrichment. 5 gives some multi-copy margin
# without discarding real signal.
MIN_OG_COUNT = 5
PADJ_THRESHOLD = 0.05

print("Loading UMAP cluster assignments...")
umap_dfs = {}
for sp, path in UMAP_FILES.items():
    df = pd.read_csv(path, usecols=['Entry', 'Cluster Label'])
    umap_dfs[sp] = df
    print(f"  {sp}: {len(df)} proteins")

print("Loading OrthoDB ortholog-group assignments...")
og_dfs = {}
og_desc = {}
for sp, path in ORTHODB_FILES.items():
    df = pd.read_csv(path, sep='\t', usecols=['#query', 'ODB_OG', 'Description'])
    df = df.rename(columns={'#query': 'Entry'}).drop_duplicates(subset=['Entry'])
    og_dfs[sp] = df
    for og, desc in zip(df['ODB_OG'], df['Description']):
        if og not in og_desc and isinstance(desc, str):
            og_desc[og] = desc

print("Joining UMAP clusters with OrthoDB groups per species...")
combined_parts = []
for sp in UMAP_FILES:
    merged = umap_dfs[sp].merge(og_dfs[sp][['Entry', 'ODB_OG']], on='Entry', how='inner')
    match_rate = len(merged) / len(umap_dfs[sp]) * 100
    print(f"  {sp}: {len(merged)}/{len(umap_dfs[sp])} proteins matched to an OG ({match_rate:.1f}%)")
    combined_parts.append(merged)
combined = pd.concat(combined_parts, ignore_index=True)
combined = combined[combined['Cluster Label'] >= 0].copy()
print(f"Total joined rows: {len(combined)}, clusters: {combined['Cluster Label'].nunique()}")

cluster_names = (pd.read_csv(list(UMAP_FILES.values())[0], usecols=['Cluster Label', 'Cluster Name'])
                  .drop_duplicates('Cluster Label')
                  .set_index('Cluster Label')['Cluster Name']
                  .to_dict())

og_counts = combined['ODB_OG'].value_counts()
vocab = og_counts[og_counts >= MIN_OG_COUNT].index
print(f"OG vocabulary (>= {MIN_OG_COUNT} occurrences): {len(vocab)} groups")

N = len(combined)
clusters = sorted(combined['Cluster Label'].unique())
cluster_totals = combined.groupby('Cluster Label').size()

print(f"Running hypergeometric tests ({len(vocab)} OGs x {len(clusters)} clusters)...")
pivot = pd.crosstab(combined['ODB_OG'], combined['Cluster Label']).loc[vocab]
og_totals = pivot.sum(axis=1)

long = pivot.stack().rename('k').reset_index()
long['K_t'] = long['ODB_OG'].map(og_totals)
long['M'] = long['Cluster Label'].map(cluster_totals)
long['pval'] = hypergeom.sf(long['k'] - 1, N, long['K_t'], long['M'])

print("Applying BH correction per OG...")
corrected = []
for og, sub in long.groupby('ODB_OG'):
    sub = sub.copy()
    _, padj, _, _ = multipletests(sub['pval'], method='fdr_bh')
    sub['padj'] = padj
    corrected.append(sub)
df_tests = pd.concat(corrected, ignore_index=True)

print("Building per-cluster results...")
result_rows = []
for cl in clusters:
    sig = df_tests[(df_tests['Cluster Label'] == cl) & (df_tests['padj'] < PADJ_THRESHOLD) & (df_tests['k'] > 0)]
    sig = sig.sort_values('padj')

    if sig.empty:
        top_og, top_desc, top_padj, all_ogs = '', '', 1.0, ''
    else:
        top_og = sig.iloc[0]['ODB_OG']
        top_desc = og_desc.get(top_og, '')
        top_padj = float(sig.iloc[0]['padj'])
        all_ogs = ';'.join(sig['ODB_OG'].tolist()[:10])

    result_rows.append({
        'Cluster Label':      cl,
        'Cluster Name':       cluster_names.get(cl, f'Cluster {cl}'),
        'Top OG':             top_og,
        'Top OG Description': top_desc,
        'Top padj':           top_padj,
        'N Enriched OGs':     len(sig),
        'All OGs':            all_ogs,
    })

df_out = pd.DataFrame(result_rows)
df_out.to_csv('umap_output/orthodb_enrichment.csv', index=False)

n_annotated = (df_out['Top OG'] != '').sum()
print(f"\nEnriched clusters: {n_annotated} / {len(clusters)} ({100*n_annotated/len(clusters):.0f}%)")
print("\nSample enrichments:")
for _, r in df_out[df_out['Top OG'] != ''].head(15).iterrows():
    print(f"  Cl {int(r['Cluster Label']):>3}  {r['Cluster Name']:<28}  -> {r['Top OG']}  ({r['Top OG Description']})  padj={r['Top padj']:.2e}")

print("\nSaved to umap_output/orthodb_enrichment.csv")
