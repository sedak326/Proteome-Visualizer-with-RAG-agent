import pandas as pd
import re
import os
from collections import defaultdict

BASE = "/home/skavlak/PLVis-Extension-with-Marine-Mammals-and-RAG"

# ── 1. Load all annotations ──────────────────────────────────────────────────
ann_files = {
    "Bottlenose Dolphin": ["bottlenose_part1_annotated.csv", "bottlenose_part2_annotated.csv"],
    "Killer Whale":       ["orca_part1_annotated.csv",       "orca_part2_annotated.csv"],
    "Gray Whale":         ["graywhale_part1_annotated.csv",  "graywhale_part2_annotated.csv"],
    "Harbor Seal":        ["harborseal_part1_annotated.csv", "harborseal_part2_annotated.csv"],
    "California Sea Lion":["sealion_part1_annotated.csv",    "sealion_part2_annotated.csv"],
}

ann_dfs = []
for species, files in ann_files.items():
    for f in files:
        path = os.path.join(BASE, "annotated", f)
        df = pd.read_csv(path, low_memory=False)
        df["_species_ann"] = species
        ann_dfs.append(df)

annotations = pd.concat(ann_dfs, ignore_index=True)
# Keep relevant columns; drop duplicate idents per species (keep first)
annotations = annotations[["ident", "desc", "hmm_labels", "_species_ann"]].copy()
annotations["desc"]       = annotations["desc"].fillna("").astype(str)
annotations["hmm_labels"] = annotations["hmm_labels"].fillna("").astype(str)

# Build ident -> {desc, hmm_labels} lookup (global; same protein may appear in
# multiple species but description should be identical – take first occurrence)
ann_lookup = {}
for _, row in annotations.iterrows():
    if row["ident"] not in ann_lookup:
        ann_lookup[row["ident"]] = {"desc": row["desc"], "hmm_labels": row["hmm_labels"]}

print(f"Total unique annotated proteins loaded: {len(ann_lookup):,}")

# ── 2. Load UMAP / cluster CSVs ──────────────────────────────────────────────
umap_files = {
    "Bottlenose Dolphin":  "Bottlenose Dolphin_umap.csv",
    "Killer Whale":        "Killer whale_umap.csv",
    "Gray Whale":          "Graywhale_umap.csv",
    "Harbor Seal":         "Harborseal_umap.csv",
    "California Sea Lion": "California Sealion_umap.csv",
}

umap_dfs = []
for species, fname in umap_files.items():
    path = os.path.join(BASE, "umap_output", fname)
    df = pd.read_csv(path, low_memory=False)
    df["_species"] = species
    umap_dfs.append(df)

umap = pd.concat(umap_dfs, ignore_index=True)
print(f"Total UMAP rows across all species: {len(umap):,}")

# Merge annotations onto umap
def lookup_desc(ident):
    r = ann_lookup.get(ident, {})
    return r.get("desc", "")

def lookup_hmm(ident):
    r = ann_lookup.get(ident, {})
    return r.get("hmm_labels", "")

umap["desc"]       = umap["Entry"].apply(lookup_desc)
umap["hmm_labels"] = umap["Entry"].apply(lookup_hmm)

# Fraction annotated
n_annotated = (umap["desc"] != "").sum()
print(f"UMAP rows with annotation: {n_annotated:,} / {len(umap):,} ({100*n_annotated/len(umap):.1f}%)")

# ── 3. Search families ───────────────────────────────────────────────────────
search_families = [
    ("PRDX / Peroxiredoxin",             r"prdx|peroxiredoxin"),
    ("Olfactory Receptor / OR / olfr",   r"olfactory\s*receptor|\bolfr\b|\bolf\b"),
    ("MHC / HLA / Histocompatibility",   r"\bmhc\b|\bhla\b|histocompatibility|major histocompat"),
    ("Immunoglobulin / IGHC / Antibody Heavy Chain", r"immunoglobulin|\bighc\b|antibody heavy|ig heavy"),
    ("Hemoglobin / Globin",              r"hemoglobin|haemoglobin|\bglobin\b"),
    ("OGT / O-GlcNAc Transferase",       r"\bogt\b|o.glcnac transferase|o-glcnac transferase"),
    ("Aquaporin / AQP",                  r"aquaporin|\baqp\b"),
    ("HIF / Hypoxia Inducible",          r"\bhif\b|hypoxia.inducible|hypoxia inducible"),
    ("FGF / Fibroblast Growth Factor",   r"\bfgf\b|fibroblast growth factor"),
    ("CYP / Cytochrome P450",            r"\bcyp\b|cytochrome p450"),
    ("CAT / Catalase",                   r"\bcat\b|catalase"),
    ("GPX / Glutathione Peroxidase",     r"\bgpx\b|glutathione peroxidase"),
    ("SOD / Superoxide Dismutase",       r"\bsod\b|superoxide dismutase"),
    ("Collagen",                         r"collagen"),
    ("Keratin",                          r"\bkeratin\b"),
]

results = []

for family_name, pattern in search_families:
    rx = re.compile(pattern, re.IGNORECASE)

    mask = umap["desc"].str.contains(rx, na=False) | umap["hmm_labels"].str.contains(rx, na=False)
    hits = umap[mask].copy()

    total      = len(hits)
    in_cluster = (hits["Cluster Label"] >= 0).sum()
    in_noise   = (hits["Cluster Label"] == -1).sum()

    cluster_ids = sorted(hits.loc[hits["Cluster Label"] >= 0, "Cluster Label"].unique().tolist())
    species_set = sorted(hits["_species"].unique().tolist())

    pct_cluster = 100 * in_cluster / total if total > 0 else 0
    pct_noise   = 100 * in_noise   / total if total > 0 else 0

    results.append({
        "Family":         family_name,
        "Total":          total,
        "In Clusters":    in_cluster,
        "% Clustered":    f"{pct_cluster:.1f}%",
        "In Noise":       in_noise,
        "% Noise":        f"{pct_noise:.1f}%",
        "Cluster IDs":    ", ".join(str(c) for c in cluster_ids) if cluster_ids else "—",
        "Species":        ", ".join(species_set) if species_set else "—",
    })

# ── 4. Print table ───────────────────────────────────────────────────────────
col_widths = {
    "Family":      40,
    "Total":        7,
    "In Clusters": 12,
    "% Clustered": 12,
    "In Noise":    10,
    "% Noise":     10,
    "Cluster IDs": 60,
    "Species":     80,
}

header = (
    f"{'Family':<40} {'Total':>7} {'Clustered':>12} {'%Clust':>12} "
    f"{'Noise':>10} {'%Noise':>10}  {'Cluster IDs':<60}  {'Species'}"
)
print("\n" + "="*len(header))
print(header)
print("="*len(header))

for r in results:
    line = (
        f"{r['Family']:<40} {r['Total']:>7} {r['In Clusters']:>12} "
        f"{r['% Clustered']:>12} {r['In Noise']:>10} {r['% Noise']:>10}  "
        f"{r['Cluster IDs']:<60}  {r['Species']}"
    )
    print(line)

print("="*len(header))

# ── 5. Also show per-species breakdown for interesting families ──────────────
print("\n\n=== PER-SPECIES BREAKDOWN ===\n")

for family_name, pattern in search_families:
    rx = re.compile(pattern, re.IGNORECASE)
    mask = umap["desc"].str.contains(rx, na=False) | umap["hmm_labels"].str.contains(rx, na=False)
    hits = umap[mask]
    if len(hits) == 0:
        continue

    print(f"--- {family_name} ---")
    for sp in sorted(hits["_species"].unique()):
        sp_hits = hits[hits["_species"] == sp]
        clust   = (sp_hits["Cluster Label"] >= 0).sum()
        noise   = (sp_hits["Cluster Label"] == -1).sum()
        ids     = sorted(sp_hits.loc[sp_hits["Cluster Label"] >= 0, "Cluster Label"].unique())
        print(f"  {sp:<26} total={len(sp_hits):>4}  clustered={clust:>4}  noise={noise:>4}  cluster_ids={ids}")
    print()
