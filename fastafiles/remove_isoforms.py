#!/usr/bin/env python3
"""Create copies of FASTA files with isoforms removed.

For each gene/protein, only the first entry is kept.
Output files are written to a 'no_isoforms/' subdirectory.
"""

import gzip
import os
import re
import sys

FASTA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(FASTA_DIR, "no_isoforms")
FASTA_EXTENSIONS = (".fasta", ".faa", ".fa", ".fasta.gz", ".faa.gz", ".fa.gz")

# Pattern to strip "isoform ..." from protein name to get the base protein
ISOFORM_RE = re.compile(r"\s+isoform\s+\S+", re.IGNORECASE)
# UniProt GN= field
GN_RE = re.compile(r"GN=(\S+)")


def open_fasta(filepath):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def get_protein_key(header):
    """Return a deduplication key for a FASTA header.

    For NCBI RefSeq headers (contain brackets like [Orcinus orca]):
        Use the protein description with the isoform label stripped.
    For UniProt headers (contain GN=):
        Use the gene name (GN field).
    Fallback:
        Use the full header (no dedup).
    """
    # NCBI RefSeq style: >XP_004262146.1 CDK-activating kinase ... isoform X3 [Orcinus orca]
    if re.search(r"\[.+\]\s*$", header):
        # Strip accession, then strip isoform label
        desc = header.split(None, 1)[1] if " " in header else header
        base = ISOFORM_RE.sub("", desc).strip()
        return base

    # UniProt style: >tr|...|... Protein name OS=... GN=GENE ...
    gn = GN_RE.search(header)
    if gn:
        return gn.group(1)

    # No pattern matched — keep the entry (unique key)
    return header


def filter_isoforms(inpath, outpath):
    """Read a FASTA file and write a filtered copy without isoform duplicates."""
    seen = set()
    kept = 0
    skipped = 0

    with open_fasta(inpath) as fin, open(outpath, "w") as fout:
        write_current = False
        for line in fin:
            if line.startswith(">"):
                key = get_protein_key(line.strip())
                if key not in seen:
                    seen.add(key)
                    write_current = True
                    kept += 1
                else:
                    write_current = False
                    skipped += 1
            if write_current:
                fout.write(line)

    return kept, skipped


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else FASTA_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f
        for f in os.listdir(target_dir)
        if any(f.endswith(ext) for ext in FASTA_EXTENSIONS)
    )

    if not files:
        print(f"No FASTA files found in {target_dir}")
        return

    for fname in files:
        inpath = os.path.join(target_dir, fname)
        # Output is always uncompressed
        out_name = fname.replace(".gz", "")
        outpath = os.path.join(out_dir, out_name)

        print(f"Processing {fname} ...", flush=True)
        kept, skipped = filter_isoforms(inpath, outpath)
        print(f"  kept {kept}, removed {skipped} isoforms -> {out_name}")

    print(f"\nFiltered files written to: {out_dir}")


if __name__ == "__main__":
    main()
