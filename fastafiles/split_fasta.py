#!/usr/bin/env python3
"""Split FASTA files so no single file exceeds a given size limit.

Splits happen at sequence boundaries (never mid-record).
Output goes to a 'split/' subdirectory.
"""

import os
import sys

MAX_BYTES = 8 * 1024 * 1024  # 8 MB
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "no_isoforms")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "split")
FASTA_EXTENSIONS = (".fasta", ".faa", ".fa")


def split_fasta(inpath, out_dir, max_bytes):
    """Split a FASTA file into parts under max_bytes each."""
    base = os.path.basename(inpath)
    name, ext = os.path.splitext(base)

    part = 1
    current_size = 0
    outfile = None
    parts = []

    def open_part():
        nonlocal outfile, current_size, part, parts
        out_name = f"{name}_part{part}{ext}"
        out_path = os.path.join(out_dir, out_name)
        outfile = open(out_path, "w")
        current_size = 0
        parts.append(out_path)

    open_part()

    with open(inpath, "r") as fin:
        record_lines = []
        for line in fin:
            if line.startswith(">") and record_lines:
                # Flush the previous record
                record = "".join(record_lines)
                record_bytes = len(record.encode("utf-8"))

                if current_size + record_bytes > max_bytes and current_size > 0:
                    outfile.close()
                    part += 1
                    open_part()

                outfile.write(record)
                current_size += record_bytes
                record_lines = []

            record_lines.append(line)

        # Flush last record
        if record_lines:
            record = "".join(record_lines)
            record_bytes = len(record.encode("utf-8"))
            if current_size + record_bytes > max_bytes and current_size > 0:
                outfile.close()
                part += 1
                open_part()
            outfile.write(record)

    if outfile:
        outfile.close()

    return parts


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else INPUT_DIR
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    max_bytes = int(sys.argv[3]) if len(sys.argv) > 3 else MAX_BYTES
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(input_dir) if any(f.endswith(e) for e in FASTA_EXTENSIONS)
    )

    if not files:
        print(f"No FASTA files found in {input_dir}")
        return

    for fname in files:
        inpath = os.path.join(input_dir, fname)
        file_size = os.path.getsize(inpath)

        if file_size <= max_bytes:
            # Just copy as-is with _part1 suffix for consistency
            out_name = f"{os.path.splitext(fname)[0]}_part1{os.path.splitext(fname)[1]}"
            out_path = os.path.join(out_dir, out_name)
            with open(inpath, "r") as fin, open(out_path, "w") as fout:
                fout.write(fin.read())
            print(f"{fname} ({file_size / 1024 / 1024:.1f} MB) -> 1 file (under limit)")
        else:
            parts = split_fasta(inpath, out_dir, max_bytes)
            sizes = [os.path.getsize(p) / 1024 / 1024 for p in parts]
            size_str = ", ".join(f"{s:.1f} MB" for s in sizes)
            print(f"{fname} ({file_size / 1024 / 1024:.1f} MB) -> {len(parts)} parts ({size_str})")

    print(f"\nSplit files written to: {out_dir}")


if __name__ == "__main__":
    main()
