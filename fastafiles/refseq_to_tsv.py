#!/usr/bin/env python3
"""
Convert RefSeq protein FASTA to UniProt-style TSV format
"""

import re
import sys
from pathlib import Path

def parse_refseq_fasta(fasta_file, organism_name, organism_code):
    """
    Parse RefSeq FASTA and extract relevant information

    RefSeq header format example:
    >XP_004270223.1 uncharacterized protein LOC101276408 [Orcinus orca]
    >XP_004270224.1 protein phosphatase 1 regulatory subunit 3A-like [Orcinus orca]
    """
    
    proteins = []
    current_entry = None
    current_sequence = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>'):
                # Save previous entry if exists
                if current_entry:
                    sequence = ''.join(current_sequence)
                    current_entry['Length'] = len(sequence)
                    current_entry['Sequence'] = sequence
                    proteins.append(current_entry)
                
                # Parse new header
                header = line[1:]  # Remove '>'
                
                # Extract accession (first part before space)
                parts = header.split(None, 1)
                accession = parts[0]
                
                # Extract protein name and gene name from description
                description = parts[1] if len(parts) > 1 else ''
                
                # Remove organism in brackets at the end
                description = re.sub(r'\s*\[.*?\]\s*$', '', description)
                
                # Try to extract gene name (often appears as "GENE-like" or just before description)
                gene_name = ''
                gene_match = re.search(r'\b([A-Z][A-Z0-9]{2,})\b', description)
                if gene_match:
                    gene_name = gene_match.group(1)
                
                # Create entry name (similar to UniProt: ACCESSION_SPECIES)
                # Extract species code from organism name
                species_code = organism_code if organism_code else 'UNKNOWN'
                entry_name = f"{accession}_{species_code}"

                current_entry = {
                    'Entry': accession,
                    'Reviewed': 'unreviewed',
                    'Entry Name': entry_name,
                    'Protein names': description,
                    'Gene Names': gene_name,
                    'Organism': organism_name,
                    'Length': 0,  # Will be calculated from sequence
                    'Annotation': '5.0'
                }
                
                current_sequence = []
            
            else:
                # Sequence line
                current_sequence.append(line)
        
        # Don't forget the last entry
        if current_entry:
            sequence = ''.join(current_sequence)
            current_entry['Length'] = len(sequence)
            current_entry['Sequence'] = sequence
            proteins.append(current_entry)
    
    return proteins

def write_tsv(proteins, output_file):
    """
    Write proteins to TSV file matching UniProt format
    """

    # Define column headers (matching bottlenose.tsv format)
    headers = ['Entry', 'Reviewed', 'Protein names', 'Gene Names', 'Entry Name', 'Length', 'Organism', 'Sequence', 'Annotation']

    with open(output_file, 'w') as f:
        # Write header
        f.write('\t'.join(headers) + '\n')

        # Write data
        for protein in proteins:
            row = [str(protein.get(h, '')) for h in headers]
            f.write('\t'.join(row) + '\n')

def main():
    if len(sys.argv) < 2:
        print("Usage: python refseq_to_tsv.py <input_fasta> [output_tsv] [organism_name] [organism_code]")
        print("\nExample:")
        print("  python refseq_to_tsv.py graywhale_protein.faa graywhale.tsv 'Eschrichtius robustus (Gray whale)' ESCRO")
        print("  python refseq_to_tsv.py input.faa output.tsv 'Orcinus orca (Killer whale)' ORCOR")
        sys.exit(1)

    input_fasta = sys.argv[1]

    # Default output filename
    if len(sys.argv) > 2:
        output_tsv = sys.argv[2]
    else:
        output_tsv = Path(input_fasta).stem + '.tsv'

    # Default organism name
    if len(sys.argv) > 3:
        organism_name = sys.argv[3]
    else:
        organism_name = 'Orcinus orca (Killer whale)'

    # Default organism code
    if len(sys.argv) > 4:
        organism_code = sys.argv[4]
    else:
        organism_code = 'ORCOR'

    print(f"Reading FASTA file: {input_fasta}")
    print(f"Organism: {organism_name}")
    print(f"Organism code: {organism_code}")
    proteins = parse_refseq_fasta(input_fasta, organism_name, organism_code)

    print(f"Found {len(proteins)} proteins")
    print(f"Writing to TSV: {output_tsv}")
    write_tsv(proteins, output_tsv)

    print("Done!")
    print(f"\nSummary:")
    print(f"  Total proteins: {len(proteins)}")
    print(f"  Output file: {output_tsv}")

if __name__ == '__main__':
    main()
