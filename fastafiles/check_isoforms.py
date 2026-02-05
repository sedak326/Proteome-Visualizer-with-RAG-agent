#!/usr/bin/env python3
"""Check FASTA files for isoform entries."""

import gzip
import os
import re
import sys
from collections import defaultdict

FASTA_DIR = os.path.dirname(os.path.abspath(__file__))
FASTA_EXTENSIONS = (".fasta", ".faa", ".fa", ".fasta.gz", ".faa.gz", ".fa.gz")


def open_fasta(filepath):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt")
    return open(filepath, "r")


def parse_headers(filepath):
    """Yield header lines from a FASTA file."""
    with open_fasta(filepath) as f:
        for line in f:
            if line.startswith(">"):
                yield line.strip()


def detect_isoforms(filepath):
    """Detect isoform entries in a FASTA file.

    Detection methods:
      1. "isoform" keyword in the header (e.g., "isoform X1", "isoform 2")
      2. UniProt isoform accessions (e.g., P12345-2)
      3. Duplicate gene names — multiple entries sharing a gene name
    """
    total = 0
    keyword_isoforms = []
    accession_isoforms = []
    gene_counts = defaultdict(list)

    # UniProt isoform accession pattern: sp|P12345-2|... or tr|A0A...-2|...
    uniprot_iso_re = re.compile(r"[sptr]{2}\|(\S+?-\d+)\|")
    # Gene name from UniProt header: GN=GENE_NAME
    uniprot_gn_re = re.compile(r"GN=(\S+)")
    # Gene name from NCBI header: extract protein name before brackets
    # Also detect "isoform Xn" pattern
    isoform_re = re.compile(r"isoform\s+\S+", re.IGNORECASE)

    for header in parse_headers(filepath):
        total += 1

        # Method 1: keyword detection
        iso_match = isoform_re.search(header)
        if iso_match:
            keyword_isoforms.append(header)

        # Method 2: UniProt accession with dash suffix (e.g., P12345-2)
        acc_match = uniprot_iso_re.search(header)
        if acc_match:
            accession_isoforms.append(header)

        # Method 3: collect gene names for duplicate detection
        gn_match = uniprot_gn_re.search(header)
        if gn_match:
            gene_counts[gn_match.group(1)].append(header)

    # Genes with multiple entries (potential isoforms even without the keyword)
    multi_gene = {g: hdrs for g, hdrs in gene_counts.items() if len(hdrs) > 1}

    return {
        "total": total,
        "keyword_isoforms": keyword_isoforms,
        "accession_isoforms": accession_isoforms,
        "multi_gene_entries": multi_gene,
    }


def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else FASTA_DIR

    files = sorted(
        f
        for f in os.listdir(target_dir)
        if any(f.endswith(ext) for ext in FASTA_EXTENSIONS)
    )

    if not files:
        print(f"No FASTA files found in {target_dir}")
        return

    for fname in files:
        filepath = os.path.join(target_dir, fname)
        results = detect_isoforms(filepath)

        print("=" * 70)
        print(f"File: {fname}")
        print(f"  Total sequences: {results['total']}")

        # Keyword isoforms
        n_kw = len(results["keyword_isoforms"])
        print(f"  Headers containing 'isoform': {n_kw}")
        if n_kw > 0:
            for h in results["keyword_isoforms"][:5]:
                print(f"    {h[:100]}")
            if n_kw > 5:
                print(f"    ... and {n_kw - 5} more")

        # Accession isoforms
        n_acc = len(results["accession_isoforms"])
        print(f"  UniProt isoform accessions (dash suffix): {n_acc}")
        if n_acc > 0:
            for h in results["accession_isoforms"][:5]:
                print(f"    {h[:100]}")
            if n_acc > 5:
                print(f"    ... and {n_acc - 5} more")

        # Multi-gene entries
        n_mg = len(results["multi_gene_entries"])
        print(f"  Genes with multiple entries: {n_mg}")
        if n_mg > 0:
            shown = 0
            for gene, hdrs in sorted(
                results["multi_gene_entries"].items(),
                key=lambda x: -len(x[1]),
            ):
                if shown >= 5:
                    print(f"    ... and {n_mg - 5} more genes")
                    break
                print(f"    {gene} ({len(hdrs)} entries)")
                shown += 1

        has_isoforms = n_kw > 0 or n_acc > 0 or n_mg > 0
        print(f"  --> Contains isoforms: {'YES' if has_isoforms else 'NO'}")
        print()


if __name__ == "__main__":
    main()
