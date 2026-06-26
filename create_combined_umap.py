# UMAPs only make sense if they are computed in the same space. 
# So this file computes a umap for all of the marine mammals we want to look at. 
# We can later separate each marine mammal UMAP if we want to look at it separately

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from sklearn.cluster import HDBSCAN
from Bio import SeqIO
import re

try:
    import cuml.manifold.umap as umap
    print("Using cuml GPU-accelerated UMAP")
    _UMAP_CLASS = umap.UMAP
except ImportError:
    import umap
    print("cuml not available, falling back to CPU umap-learn")
    _UMAP_CLASS = umap.UMAP

# FASTA headers store multiple pieces of metadata in a single string.
# This function extracts and structures that metadata so it can be
# reliably used for filtering, labeling and visualization..
def parse_fasta_header(header):
    """Parse FASTA header to extract Entry, Protein names, Gene Names, Entry Name, Organism.

    Handles two formats:
      UniProt: >sp|P12345|GENE_HUMAN Protein name OS=Homo sapiens OX=9606 GN=GENE PE=1 SV=1
      RefSeq:  >NP_001234.1 protein description [Organism name]
    """
    entry = header.split()[0].replace('>', '')

    # UniProt format uses OS= for organism; RefSeq uses [Organism]
    if 'OS=' in header:
        # Organism: between OS= and the next two-letter field (OX=, GN=, PE=, SV=) or end
        os_match = re.search(r'OS=(.*?)(?:\s+[A-Z]{2}=|\s*$)', header)
        organism = os_match.group(1).strip() if os_match else ''

        # Protein name: between the entry ID token and OS=
        pn_match = re.search(r'>\S+\s+(.*?)\s+OS=', header)
        protein_name = pn_match.group(1).strip() if pn_match else ''

        # Gene name from GN=
        gn_match = re.search(r'GN=(\S+)', header)
        gene_names = gn_match.group(1) if gn_match else ''
    else:
        # RefSeq: organism in [brackets]
        organism_match = re.search(r'\[(.*?)\]', header)
        organism = organism_match.group(1) if organism_match else ''

        pn_match = re.search(r'>\S+\s+(.*?)\s+\[', header)
        protein_name = pn_match.group(1).strip() if pn_match else ''

        gene_names = ''

    entry_name = entry

    return {
        'Entry': entry,
        'Protein names': protein_name,
        'Gene Names': gene_names,
        'Entry Name': entry_name,
        'Organism': organism
    }

# Sequences and metadata are needed separately for downstream tasks
# (e.g., embeddings, UMAP visualization, clustering, and annotation).
# This function loads both from a FASTA file and ensures they stay aligned
# via a shared, stable key.
def load_fasta_sequences(fasta_path):
    """Load sequences from FASTA file"""
    sequences = {}
    metadata = {}

    for record in SeqIO.parse(fasta_path, 'fasta'):
        key = record.id.replace('.', '_') + ' ' + record.description[len(record.id)+1:]
        sequences[key] = str(record.seq)

        full_header = '>' + record.description
        metadata[key] = parse_fasta_header(full_header)

    return sequences, metadata


# Protein embeddings are precomputed and stored to avoid
# recomputation. This function loads them into memory in a format that
# can be directly used for UMAP and clustering.
def load_embeddings_from_h5(h5_path):
    """Load embeddings from h5 file"""
    embeddings = {}

    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            embeddings[key] = np.array(f[key])

    return embeddings

# Each organism has sequences and embeddings stored separately.
# This function loads both, matches them by key, and attaches metadata
# and organism labels so all species can later be combined into a
# shared embedding space.
def load_single_organism_data(h5_path, fasta_path, organism_name):
    """Load data for a single organism"""
    print(f"Loading {organism_name}...", flush=True)

    # Load sequences and metadata from FASTA
    sequences, metadata = load_fasta_sequences(fasta_path)
    print(f"  Loaded {len(sequences)} sequences from FASTA", flush=True)

    # Load embeddings from h5
    embeddings_dict = load_embeddings_from_h5(h5_path)
    print(f"  Loaded {len(embeddings_dict)} embeddings from h5", flush=True)

    # Match sequences with embeddings
    matched_data = []
    embeddings_list = []

    for key in embeddings_dict.keys():
        if key in sequences:
            matched_data.append({
                **metadata[key],
                'Sequence': sequences[key],
                'Length': len(sequences[key]),
                'organism_label': organism_name  # Add label for tracking
            })
            embeddings_list.append(embeddings_dict[key])

    print(f"  Matched {len(matched_data)} sequences with embeddings", flush=True)

    if len(matched_data) == 0:
        print(f"  ERROR: No matches found!", flush=True)
        return None, None

    return matched_data, embeddings_list


# UMAP projections are only comparable if they are computed in the
# same embedding space. This main pipeline therefore:
# 1) Loads all species together
# 2) Computes a single shared UMAP
# 3) Clusters the combined dataset
# 4) Saves per-species subsets that remain directly comparable
def main():
    # Define all organisms to process
    organisms = [
        {
            'h5': Path('h5py_files/bottlenose.h5'),
            'fasta': Path('fastafiles/no_isoforms/bottlenose.fasta'),
            'name': 'Tursiops truncatus',
            'short_name': 'Bottlenose Dolphin'
        },
        {
            'h5': Path('h5py_files/graywhale.h5'),
            'fasta': Path('fastafiles/no_isoforms/graywhale.faa'),
            'name': 'Eschrichtius robustus',
            'short_name': 'Graywhale'
        },
        {
            'h5': Path('h5py_files/harborseal.h5'),
            'fasta': Path('fastafiles/no_isoforms/harborseal.faa'),
            'name': 'Phoca vitulina',
            'short_name': 'Harborseal'
        },
        {
            'h5': Path('h5py_files/orca.h5'),
            'fasta': Path('fastafiles/no_isoforms/orca.faa'),
            'name': 'Orcinus orca',
            'short_name': 'Killer whale'
        },
        {
            'h5': Path('h5py_files/sealion.h5'),
            'fasta': Path('fastafiles/no_isoforms/sealion.fasta'),
            'name': 'Zalophus californianus',
            'short_name': 'California Sealion'
        },
        {
            'h5': Path('h5py_files/polarbear.h5'),
            'fasta': Path('fastafiles/no_isoforms/uniprotkb_proteome_UP000261680_2026_06_17.fasta'),
            'name': 'Ursus maritimus',
            'short_name': 'Polar Bear'
        },
        {
            'h5': Path('h5py_files/hippo.h5'),
            'fasta': Path('fastafiles/no_isoforms/GCF_030028045.1_mHipAmp2.hap2_protein.faa'),
            'name': 'Hippopotamus amphibius',
            'short_name': 'Hippopotamus'
        },
    ]

    # Step 1: Load all organism data
    print("=" * 60, flush=True)
    print("STEP 1: Loading all organism data...", flush=True)
    print("=" * 60, flush=True)

    all_metadata = []
    all_embeddings = []
    organism_indices = []  # Track which organism each protein belongs to

    for i, org_info in enumerate(organisms):
        if not org_info['h5'].exists():
            print(f"WARNING: {org_info['h5']} not found, skipping...", flush=True)
            continue
        if not org_info['fasta'].exists():
            print(f"WARNING: {org_info['fasta']} not found, skipping...", flush=True)
            continue

        metadata, embeddings = load_single_organism_data(
            org_info['h5'],
            org_info['fasta'],
            org_info['name']
        )

        if metadata is not None:
            all_metadata.extend(metadata)
            all_embeddings.extend(embeddings)
            organism_indices.extend([i] * len(metadata))
            print(f"  Total proteins so far: {len(all_metadata)}\n", flush=True)

    if len(all_metadata) == 0:
        print("ERROR: No data loaded from any organism!", flush=True)
        return

    # Convert to numpy float32 — cuml requires float32, and it's half the memory of float64
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"\n{'=' * 60}", flush=True)
    print(f"Combined embeddings shape: {embeddings_array.shape}", flush=True)
    print(f"Total proteins from all species: {len(all_metadata)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # Step 2: Compute UMAP on ALL data together
    print("STEP 2: Computing UMAP on combined dataset...", flush=True)
    print("This creates a shared embedding space for all species", flush=True)
    print("-" * 60, flush=True)

    # n_neighbors=50 captures global protein-family structure for large cross-species datasets;
    # n_neighbors=15 (default) only sees local micro-structure and produces a continuous cloud.
    # min_dist=0.0 tightens clusters so K-means finds real groups instead of diffuse blobs.
    reducer = _UMAP_CLASS(n_components=2, random_state=42, n_neighbors=50, min_dist=0.0)
    embedding_2d = np.array(reducer.fit_transform(embeddings_array))  # np.array() handles cupy output
    print(f"UMAP embedding complete!", flush=True)
    print(f"Embedding shape: {embedding_2d.shape}", flush=True)
    print(f"UMAP 1 range: [{embedding_2d[:, 0].min():.2f}, {embedding_2d[:, 0].max():.2f}]", flush=True)
    print(f"UMAP 2 range: [{embedding_2d[:, 1].min():.2f}, {embedding_2d[:, 1].max():.2f}]", flush=True)
    print(flush=True)

    # Step 3: Cluster on 2D UMAP coordinates so clusters match visible blobs
    print("STEP 3: Clustering combined dataset...", flush=True)
    print("-" * 60, flush=True)
    print("Using HDBSCAN on UMAP 2D coordinates (clusters will match visual groups)", flush=True)

    hdbscan = HDBSCAN(min_cluster_size=150, min_samples=15)
    cluster_labels = hdbscan.fit_predict(embedding_2d)

    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"Clustering complete! Found {n_clusters_found} clusters, {n_noise} unclustered points (label=-1)\n", flush=True)

    # Step 4: Create and save individual organism files
    print("STEP 4: Saving individual organism files...", flush=True)
    print("=" * 60, flush=True)

    output_dir = Path('umap_output')
    output_dir.mkdir(exist_ok=True)

    for i, org_info in enumerate(organisms):
        # Get indices for this organism
        org_mask = np.array(organism_indices) == i

        if not org_mask.any():
            continue

        # Extract data for this organism
        org_metadata = [all_metadata[j] for j in range(len(all_metadata)) if organism_indices[j] == i]
        org_embeddings = embeddings_array[org_mask]
        org_umap = embedding_2d[org_mask]
        org_clusters = cluster_labels[org_mask]

        # Create dataframe
        output_df = pd.DataFrame(org_metadata)

        # Add Annotation column
        output_df['Annotation'] = 5.0

        # Add Embeddings column
        output_df['Embeddings'] = [str(emb) for emb in org_embeddings]

        # Add UMAP coordinates (from shared embedding space!)
        output_df['UMAP 1'] = org_umap[:, 0]
        output_df['UMAP 2'] = org_umap[:, 1]

        # Add Cluster Label
        output_df['Cluster Label'] = org_clusters

        # Reorder columns
        column_order = ['Entry', 'Protein names', 'Gene Names', 'Entry Name',
                       'Length', 'Organism', 'Sequence', 'Annotation',
                       'Embeddings', 'UMAP 1', 'UMAP 2', 'Cluster Label']
        output_df = output_df[column_order]

        # Save to CSV
        output_path = output_dir / f"{org_info['short_name']}_umap.csv"
        output_df.to_csv(output_path, index=False)

        print(f"{org_info['short_name']}:", flush=True)
        print(f"  Saved: {output_path}", flush=True)
        print(f"  Shape: {output_df.shape}", flush=True)
        print(f"  UMAP 1 range: [{output_df['UMAP 1'].min():.2f}, {output_df['UMAP 1'].max():.2f}]", flush=True)
        print(f"  UMAP 2 range: [{output_df['UMAP 2'].min():.2f}, {output_df['UMAP 2'].max():.2f}]", flush=True)
        print(f"  Clusters: {output_df['Cluster Label'].nunique()}", flush=True)
        print(flush=True)

    print("=" * 60, flush=True)
    print("DONE! All species now share the same UMAP embedding space", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
