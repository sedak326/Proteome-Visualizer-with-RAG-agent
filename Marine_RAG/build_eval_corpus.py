#!/usr/bin/env python3
"""
build_eval_corpus.py — Export PLVis ChromaDB chunks to AutoRAG eval parquets.

Extracts all 2,628 chunks from the marine_rag_v1 Chroma collection and writes
two files to eval/:

  corpus_plvis.parquet  — one row per chunk (AutoRAG Corpus format)
      doc_id         — ChromaDB UUID
      contents       — chunk text (includes GPT-4o-mini image descriptions)
      path           — source file path (derived from document_name metadata)
      start_end_idx  — [0, len(contents)] (offsets not stored in ChromaDB)
      metadata       — full metadata dict from ChromaDB

  raw_plvis.parquet     — one row per source document (AutoRAG Raw format)
      texts          — all chunks for that document concatenated
      path           — source file path
      page           — -1 (page-level splitting not used)
      last_modified_datetime — today's date (not stored in ChromaDB)

The Raw is built from the corpus so that GPT-4o-mini image descriptions are
included — they only exist in the ChromaDB chunks, not in the original files.

Usage:
    cd PLVis-Extension-with-Marine-Mammals-and-RAG/Marine_RAG
    python build_eval_corpus.py

Requires:
    OPENAI_API_KEY set in environment (used to initialise the embedding function)
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR  = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME     = "marine_rag_v1"
EMBEDDING_MODEL     = "text-embedding-3-small"
DATA_DIR            = Path(__file__).resolve().parent / "data"
OUTPUT_DIR          = Path(__file__).resolve().parent / "eval"
CORPUS_OUTPUT_PATH  = OUTPUT_DIR / "corpus_plvis.parquet"
RAW_OUTPUT_PATH     = OUTPUT_DIR / "raw_plvis.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(document_name: str, data_dir: Path) -> str:
    """
    Try to match document_name back to an actual file in data_dir.

    ChromaDB metadata stores document_name as either:
      - The chapter title for HTML files  (e.g. 'Abundance Estimation')
      - A sanitised PDF stem              (e.g. '1471 2148 9 20')

    We scan data_dir once and pick the closest match by checking whether the
    document_name tokens appear in the filename. Falls back to the raw name.
    """
    if not hasattr(_resolve_path, "_index"):
        # Build a lookup: lower-cased filename stem → actual filename
        _resolve_path._index = {
            f.stem.lower().replace("-", " ").replace("_", " "): f.name
            for f in data_dir.iterdir()
            if f.is_file()
        }

    normalised = document_name.lower().replace("-", " ").replace("_", " ")
    if normalised in _resolve_path._index:
        return f"data/{_resolve_path._index[normalised]}"

    # Partial match: pick the first filename whose stem contains all tokens
    tokens = normalised.split()
    for stem, fname in _resolve_path._index.items():
        if all(t in stem for t in tokens):
            return f"data/{fname}"

    return f"data/{document_name}"  # fallback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_eval_corpus() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Chroma collection '{COLLECTION_NAME}' from {CHROMA_PERSIST_DIR} …")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vs = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    col = vs._collection
    total = col.count()
    print(f"  {total} chunks found")

    print("Fetching all chunks …")
    result = col.get(include=["documents", "metadatas"])

    # ------------------------------------------------------------------
    # Build corpus_plvis.parquet — one row per chunk
    # ------------------------------------------------------------------
    corpus_rows = []
    for doc_id, content, meta in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        path = _resolve_path(meta.get("document_name", ""), DATA_DIR)
        corpus_rows.append(
            {
                "doc_id":        doc_id,
                "contents":      content,
                "path":          path,
                "start_end_idx": [0, len(content)],
                "metadata":      meta,
            }
        )

    corpus_df = pd.DataFrame(corpus_rows)
    corpus_df.to_parquet(CORPUS_OUTPUT_PATH, index=False)
    print(f"\nSaved {len(corpus_df)} chunks → {CORPUS_OUTPUT_PATH}")

    unique_paths = corpus_df["path"].nunique()
    print(f"  {unique_paths} unique source files referenced")
    content_types = {}
    for meta in result["metadatas"]:
        ct = meta.get("content_type", "unknown")
        content_types[ct] = content_types.get(ct, 0) + 1
    for ct, count in sorted(content_types.items()):
        print(f"  {ct}: {count} chunks")

    # ------------------------------------------------------------------
    # Build raw_plvis.parquet — one row per source document
    # Concatenate all chunk texts per path so image descriptions are
    # included (they only exist in the ChromaDB chunks, not on disk).
    # ------------------------------------------------------------------
    today = str(date.today())
    raw_rows = (
        corpus_df.groupby("path")["contents"]
        .apply(lambda chunks: "\n\n".join(chunks))
        .reset_index()
        .rename(columns={"contents": "texts"})
    )
    raw_rows["page"] = -1
    raw_rows["last_modified_datetime"] = today

    raw_rows.to_parquet(RAW_OUTPUT_PATH, index=False)
    print(f"\nSaved {len(raw_rows)} source documents → {RAW_OUTPUT_PATH}")


if __name__ == "__main__":
    build_eval_corpus()
