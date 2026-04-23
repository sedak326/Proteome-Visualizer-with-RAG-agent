#!/usr/bin/env python3
"""
build_eval_corpus_ollama.py — Export ChromaDB chunks to AutoRAG eval parquets.

Writes two files to eval/:

  corpus_plvis.parquet  — one row per chunk (AutoRAG Corpus format)
  raw_plvis.parquet     — one row per source document (AutoRAG Raw format)

Usage:
    python build_eval_corpus_ollama.py

Requires:
    pip install langchain-community langchain-ollama chromadb pandas pyarrow
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR  = Path("/workspace/RAGapplication_multiPDF/eval/chroma_db")
COLLECTION_NAME     = "paypal_docs_v1"
DATA_DIR            = Path("/workspace/RAGapplication_multiPDF/data")
OUTPUT_DIR          = Path("/workspace/RAGapplication_multiPDF/eval")
CORPUS_OUTPUT_PATH  = OUTPUT_DIR / "corpus_plvis.parquet"
RAW_OUTPUT_PATH     = OUTPUT_DIR / "raw_plvis.parquet"
OLLAMA_URL          = "http://localhost:11434"
EMBEDDING_MODEL     = "nomic-embed-text"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(document_name: str, data_dir: Path) -> str:
    if not hasattr(_resolve_path, "_index"):
        _resolve_path._index = {
            f.stem.lower().replace("-", " ").replace("_", " "): f.name
            for f in data_dir.iterdir()
            if f.is_file()
        }

    normalised = document_name.lower().replace("-", " ").replace("_", " ")
    if normalised in _resolve_path._index:
        return f"data/{_resolve_path._index[normalised]}"

    tokens = normalised.split()
    for stem, fname in _resolve_path._index.items():
        if all(t in stem for t in tokens):
            return f"data/{fname}"

    return f"data/{document_name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_eval_corpus() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Chroma collection '{COLLECTION_NAME}' from {CHROMA_PERSIST_DIR} …")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
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
