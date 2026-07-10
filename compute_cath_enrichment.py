"""
Compute per-cluster CATH structural/functional enrichment (evolutionary relationship
enrichment via structure, complementing the OrthoDB ortholog-group enrichment).

For each K-means cluster, runs a hypergeometric test (+ BH correction) against every
CATH Gene3D structural superfamily and every CATH-Gene3D FunFam (functional
sub-family within a superfamily) present in the dataset -- the same kind of test the
original PLVis paper ran against CATH FunFams. Gene3D is the coarser structural
classification; FunFam is the finer functional split used by the original paper.

Requires results/cath/<species>.tsv, produced by InterProScan with -appl Gene3D,FunFam
(see /uss/skavlak/plvis_homology_enrichment/run_cath_all.sh). Standard 15-column
InterProScan TSV, no header: col1=protein id, col4=analysis (Gene3D/FunFam),
col5=signature accession, col6=signature description.

Saves umap_output/cath_enrichment.csv with columns:
  Cluster Label, Cluster Name, Analysis, Top Signature, Top Signature Description,
  Top padj, N Enriched Signatures, All Signatures (semicolon-sep)
one row per (cluster, analysis) pair, i.e. two rows per cluster (Gene3D and FunFam).
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

CATH_FILES = {
    'Sea Lion':    '/uss/skavlak/plvis_homology_enrichment/results/cath/sealion.tsv',
    'Bottlenose':  '/uss/skavlak/plvis_homology_enrichment/results/cath/bottlenose.tsv',
    'Gray Whale':  '/uss/skavlak/plvis_homology_enrichment/results/cath/graywhale.tsv',
    'Orca':        '/uss/skavlak/plvis_homology_enrichment/results/cath/orca.tsv',
    'Harbor Seal': '/uss/skavlak/plvis_homology_enrichment/results/cath/harborseal.tsv',
    'Polar Bear':  '/uss/skavlak/plvis_homology_enrichment/results/cath/polarbear.tsv',
    'Hippo':       '/uss/skavlak/plvis_homology_enrichment/results/cath/hippo.tsv',
}

IPRSCAN_COLS = ['Entry', 'MD5', 'Length', 'Analysis', 'Signature', 'SigDescription',
                'Start', 'Stop', 'Score', 'Status', 'Date', 'InterPro', 'InterProDesc',
                'GO', 'Pathways']

MIN_SIG_COUNT = 5   # ignore signatures observed fewer than this many times across the dataset
PADJ_THRESHOLD = 0.05

print("Loading UMAP cluster assignments...")
umap_dfs = {}
for sp, path in UMAP_FILES.items():
    df = pd.read_csv(path, usecols=['Entry', 'Cluster Label'])
    umap_dfs[sp] = df
    print(f"  {sp}: {len(df)} proteins")

print("Loading CATH Gene3D/FunFam assignments...")
cath_dfs = {}
sig_desc = {}
for sp, path in CATH_FILES.items():
    df = pd.read_csv(path, sep='\t', header=None, names=IPRSCAN_COLS,
                      usecols=['Entry', 'Analysis', 'Signature', 'SigDescription'])
    df = df.drop_duplicates(subset=['Entry', 'Analysis', 'Signature'])
    cath_dfs[sp] = df
    for sig, desc in zip(df['Signature'], df['SigDescription']):
        if sig not in sig_desc and isinstance(desc, str) and desc != '-':
            sig_desc[sig] = desc

cluster_names = (pd.read_csv(list(UMAP_FILES.values())[0], usecols=['Cluster Label', 'Cluster Name'])
                  .drop_duplicates('Cluster Label')
                  .set_index('Cluster Label')['Cluster Name']
                  .to_dict())

all_results = []

for analysis in ['Gene3D', 'FunFam']:
    print(f"\n=== {analysis} ===")
    print("Joining UMAP clusters with signatures per species...")
    combined_parts = []
    for sp in UMAP_FILES:
        sub = cath_dfs[sp][cath_dfs[sp]['Analysis'] == analysis][['Entry', 'Signature']]
        merged = umap_dfs[sp].merge(sub, on='Entry', how='inner')
        match_rate = len(merged['Entry'].unique()) / len(umap_dfs[sp]) * 100
        print(f"  {sp}: {merged['Entry'].nunique()}/{len(umap_dfs[sp])} proteins matched ({match_rate:.1f}%)")
        combined_parts.append(merged)
    combined = pd.concat(combined_parts, ignore_index=True)
    combined = combined[combined['Cluster Label'] >= 0].copy()
    print(f"Total joined rows: {len(combined)}, clusters: {combined['Cluster Label'].nunique()}")

    sig_counts = combined['Signature'].value_counts()
    vocab = sig_counts[sig_counts >= MIN_SIG_COUNT].index
    print(f"Signature vocabulary (>= {MIN_SIG_COUNT} occurrences): {len(vocab)}")

    N = len(combined)
    clusters = sorted(combined['Cluster Label'].unique())
    cluster_totals = combined.groupby('Cluster Label').size()

    print(f"Running hypergeometric tests ({len(vocab)} signatures x {len(clusters)} clusters)...")
    pivot = pd.crosstab(combined['Signature'], combined['Cluster Label']).loc[vocab]
    sig_totals = pivot.sum(axis=1)

    long = pivot.stack().rename('k').reset_index()
    long['K_t'] = long['Signature'].map(sig_totals)
    long['M'] = long['Cluster Label'].map(cluster_totals)
    long['pval'] = hypergeom.sf(long['k'] - 1, N, long['K_t'], long['M'])

    print("Applying BH correction per signature...")
    corrected = []
    for sig, sub in long.groupby('Signature'):
        sub = sub.copy()
        _, padj, _, _ = multipletests(sub['pval'], method='fdr_bh')
        sub['padj'] = padj
        corrected.append(sub)
    df_tests = pd.concat(corrected, ignore_index=True)

    for cl in clusters:
        sig_hits = df_tests[(df_tests['Cluster Label'] == cl) & (df_tests['padj'] < PADJ_THRESHOLD) & (df_tests['k'] > 0)]
        sig_hits = sig_hits.sort_values('padj')

        if sig_hits.empty:
            top_sig, top_desc, top_padj, all_sigs = '', '', 1.0, ''
        else:
            top_sig = sig_hits.iloc[0]['Signature']
            top_desc = sig_desc.get(top_sig, '')
            top_padj = float(sig_hits.iloc[0]['padj'])
            all_sigs = ';'.join(sig_hits['Signature'].tolist()[:10])

        all_results.append({
            'Cluster Label':              cl,
            'Cluster Name':               cluster_names.get(cl, f'Cluster {cl}'),
            'Analysis':                   analysis,
            'Top Signature':              top_sig,
            'Top Signature Description':  top_desc,
            'Top padj':                   top_padj,
            'N Enriched Signatures':      len(sig_hits),
            'All Signatures':             all_sigs,
        })

df_out = pd.DataFrame(all_results)
df_out.to_csv('umap_output/cath_enrichment.csv', index=False)

for analysis in ['Gene3D', 'FunFam']:
    sub = df_out[df_out['Analysis'] == analysis]
    n_annotated = (sub['Top Signature'] != '').sum()
    print(f"\n{analysis} enriched clusters: {n_annotated} / {len(sub)} ({100*n_annotated/len(sub):.0f}%)")

print("\nSample enrichments:")
for _, r in df_out[df_out['Top Signature'] != ''].head(15).iterrows():
    print(f"  Cl {int(r['Cluster Label']):>3}  {r['Cluster Name']:<28}  [{r['Analysis']}]  -> {r['Top Signature']}  ({r['Top Signature Description']})  padj={r['Top padj']:.2e}")

print("\nSaved to umap_output/cath_enrichment.csv")
