"""
Run ProtT5 embeddings for polar bear and hippo.
Uses GPU 0. Writes to h5py_files/.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from h5py_files.prott5_embedder import get_embeddings
from pathlib import Path

BASE = Path(__file__).parent

JOBS = [
    {
        'fasta': BASE / 'fastafiles/no_isoforms/uniprotkb_proteome_UP000261680_2026_06_17.fasta',
        'h5':    BASE / 'h5py_files/polarbear.h5',
        'name':  'Polar Bear',
    },
    {
        'fasta': BASE / 'fastafiles/no_isoforms/GCF_030028045.1_mHipAmp2.hap2_protein.faa',
        'h5':    BASE / 'h5py_files/hippo.h5',
        'name':  'Hippopotamus',
    },
]

model_dir = None  # uses HuggingFace cache

for job in JOBS:
    print(f"\n{'='*60}")
    print(f"Embedding: {job['name']}")
    print(f"  FASTA: {job['fasta']}")
    print(f"  Output: {job['h5']}")
    print('='*60, flush=True)
    if job['h5'].exists():
        print(f"  Already exists, skipping.")
        continue
    get_embeddings(
        seq_path=job['fasta'],
        emb_path=job['h5'],
        model_dir=model_dir,
        per_protein=True,
    )
    print(f"  Done: {job['name']}", flush=True)

print("\nAll embeddings complete.")
