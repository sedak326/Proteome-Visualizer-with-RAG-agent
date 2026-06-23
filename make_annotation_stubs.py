"""
Create minimal annotation CSVs for new species (polar bear, hippo).
Populates ident/sequence/desc/length/taxonomy; leaves HMM fields empty.
These match the schema expected by load_annotations() in the app.
"""
import gzip, re, os, csv, math
from pathlib import Path

BASE = Path(__file__).parent

COLS = [
    'ident', 'sequence', 'desc', 'clipID', 'clipDescription',
    'genus', 'family', 'order', 'class', 'phylum', 'domain',
    'length', 'hmm_labels', 'hmm_pfam_ids', 'hmm_ranges',
    'hmm_descriptions', 'clipScore', 'percentIdentity',
    'queryCoverage', 'annotationConfidence', 'species',
]

TAXONOMY = {
    'polarbear': dict(
        genus='Ursus', family='Ursidae', order='Carnivora',
        cls='Mammalia', phylum='Chordata', domain='Eukaryota',
        species='Ursus maritimus',
    ),
    'hippo': dict(
        genus='Hippopotamus', family='Hippopotamidae', order='Artiodactyla',
        cls='Mammalia', phylum='Chordata', domain='Eukaryota',
        species='Hippopotamus amphibius',
    ),
}

SOURCES = {
    'polarbear': BASE / 'fastafiles/no_isoforms/uniprotkb_proteome_UP000261680_2026_06_17.fasta',
    'hippo':     BASE / 'fastafiles/no_isoforms/GCF_030028045.1_mHipAmp2.hap2_protein.faa',
}

OUT_DIR = BASE / 'annotated'

def parse_fasta(path):
    """Yields (header, sequence) pairs from plain FASTA."""
    cur_id, cur_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if cur_id:
                    yield cur_id, ''.join(cur_seq)
                cur_id = line[1:]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id:
        yield cur_id, ''.join(cur_seq)


def make_ident(header):
    """Use just the FASTA accession (first word), matching the Entry column in UMAP CSVs."""
    return header.split()[0]


for key, fasta_path in SOURCES.items():
    tax = TAXONOMY[key]
    records = list(parse_fasta(fasta_path))
    n = len(records)
    mid = math.ceil(n / 2)
    parts = [records[:mid], records[mid:]]

    for i, chunk in enumerate(parts, 1):
        out_path = OUT_DIR / f'{key}_part{i}_annotated.csv'
        if out_path.exists():
            print(f'  {out_path.name}: already exists, skipping.')
            continue

        with open(out_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=COLS)
            writer.writeheader()
            for header, seq in chunk:
                ident = make_ident(header)
                writer.writerow({
                    'ident':               ident,
                    'sequence':            seq,
                    'desc':                header,
                    'clipID':              '',
                    'clipDescription':     '',
                    'genus':               tax['genus'],
                    'family':              tax['family'],
                    'order':               tax['order'],
                    'class':               tax['cls'],
                    'phylum':              tax['phylum'],
                    'domain':              tax['domain'],
                    'length':              len(seq),
                    'hmm_labels':          '',
                    'hmm_pfam_ids':        '',
                    'hmm_ranges':          '',
                    'hmm_descriptions':    '',
                    'clipScore':           '',
                    'percentIdentity':     '',
                    'queryCoverage':       '',
                    'annotationConfidence':'',
                    'species':             tax['species'],
                })
        print(f'  Wrote {len(chunk):,} records → {out_path.name}')

print('Done.')
