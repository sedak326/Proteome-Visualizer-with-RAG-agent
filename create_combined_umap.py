# UMAPs only make sense if they are computed in the same space. 
# So this file computes a umap for all of the marine mammals we want to look at. 
# We can later separate each marine mammal UMAP if we want to look at it separately

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
import umap
from sklearn.cluster import KMeans
from Bio import SeqIO
import re

# FASTA headers store multiple pieces of metadata in a single string.
# This function extracts and structures that metadata so it can be
# reliably used for filtering, labeling and visualization..
def parse_fasta_header(header):
    """Parse FASTA header to extract Entry, Protein names, Gene Names, Entry Name, Organism"""
    entry = header.split()[0].replace('>', '')

    organism_match = re.search(r'\[(.*?)\]', header)
    organism = organism_match.group(1) if organism_match else ''

    protein_name_match = re.search(r'>\S+\s+(.*?)\s+\[', header)
    protein_name = protein_name_match.group(1).strip() if protein_name_match else ''

    entry_name = entry
    gene_names = ''

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


# Protein embeddings are precomputed and stored on disk to avoid
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
            'h5': Path('h5py_files/bottlenose_UP000245320_2025_10_30.h5'),
            'fasta': Path('fastafiles/bottlenose_UP000245320_2025_10_30.fasta'),
            'name': 'Tursiops truncatus',
            'short_name': 'bottlenose'
        },
        {
            'h5': Path('h5py_files/graywhale_proteome_UP001159641_2025_10_30.h5'),
            'fasta': Path('fastafiles/graywhale_protein.faa'),
            'name': 'Eschrichtius robustus',
            'short_name': 'graywhale'
        },
        {
            'h5': Path('h5py_files/harborseal.h5'),
            'fasta': Path('fastafiles/harborseal.faa'),
            'name': 'Phoca vitulina',
            'short_name': 'harborseal'
        },
        {
            'h5': Path('h5py_files/orca.h5'),
            'fasta': Path('fastafiles/orca.faa'),
            'name': 'Orcinus orca',
            'short_name': 'orca'
        },
        {
            'h5': Path('h5py_files/sealion_UP000515165_2025_10_30.h5'),
            'fasta': Path('fastafiles/sealion_UP000515165_2025_10_30.fasta'),
            'name': 'Zalophus californianus',
            'short_name': 'sealion'
        }
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

    # Convert to numpy array
    embeddings_array = np.array(all_embeddings)
    print(f"\n{'=' * 60}", flush=True)
    print(f"Combined embeddings shape: {embeddings_array.shape}", flush=True)
    print(f"Total proteins from all species: {len(all_metadata)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    # Step 2: Compute UMAP on ALL data together
    print("STEP 2: Computing UMAP on combined dataset...", flush=True)
    print("This creates a shared embedding space for all species", flush=True)
    print("-" * 60, flush=True)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings_array)
    print(f"UMAP embedding complete!", flush=True)
    print(f"Embedding shape: {embedding_2d.shape}", flush=True)
    print(f"UMAP 1 range: [{embedding_2d[:, 0].min():.2f}, {embedding_2d[:, 0].max():.2f}]", flush=True)
    print(f"UMAP 2 range: [{embedding_2d[:, 1].min():.2f}, {embedding_2d[:, 1].max():.2f}]", flush=True)
    print(flush=True)

    # Step 3: Perform clustering on combined data
    print("STEP 3: Clustering combined dataset...", flush=True)
    print("-" * 60, flush=True)

    n_clusters = min(100, len(all_metadata) // 50)
    if n_clusters < 2:
        n_clusters = 2

    print(f"Using {n_clusters} clusters", flush=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    print(f"Clustering complete!\n", flush=True)

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
