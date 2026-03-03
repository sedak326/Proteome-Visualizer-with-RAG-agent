#!/usr/bin/env python3
"""
build_index.py — Build Chroma vector store from corpus.jsonl

Run this once after extract_corpus.py, and re-run whenever the corpus changes.

Usage:
    python build_index.py

Requires:
    pip install langchain-chroma langchain-openai openai
    OPENAI_API_KEY set in environment
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JSONL_PATH         = Path("project/corpus.jsonl")
CHROMA_PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME    = "marine_rag_v1"
EMBEDDING_MODEL    = "text-embedding-3-small"
BATCH_SIZE         = 50   # OpenAI embeddings API handles larger batches than Ollama


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_documents(jsonl_path: Path) -> List[Document]:
    docs: List[Document] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = (item.get("text") or "").strip()
            if not text:
                continue
            metadata = {k: v for k, v in item.items() if k != "text"}
            metadata.setdefault("content_type", "text")
            # Chroma requires flat scalar metadata — stringify any nested values
            for key in list(metadata.keys()):
                if isinstance(metadata[key], (dict, list)):
                    metadata[key] = json.dumps(metadata[key])
                elif metadata[key] is None:
                    metadata[key] = ""
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_index() -> None:
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"{JSONL_PATH} not found — run extract_corpus.py first."
        )

    print(f"Loading documents from {JSONL_PATH}…")
    docs = load_documents(JSONL_PATH)
    docs = filter_complex_metadata(docs)
    counts = Counter(d.metadata.get("content_type") for d in docs)
    print(f"  {len(docs)} documents loaded: {dict(counts)}")

    print(f"\nInitialising embeddings ({EMBEDDING_MODEL})…")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    # Sanity check
    test = embeddings.embed_query("test")
    print(f"  Embedding dim: {len(test)}")

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nIndexing into Chroma ({CHROMA_PERSIST_DIR} / {COLLECTION_NAME})…")
    start = time.time()
    vectorstore = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_PERSIST_DIR),
                collection_name=COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)
        done = min(i + BATCH_SIZE, len(docs))
        print(f"  [{done}/{len(docs)}]", end="\r", flush=True)

    elapsed = time.time() - start
    total = vectorstore._collection.count()
    print(f"\n  Done in {elapsed:.1f}s  —  {total} vectors stored")
    print(f"\nChroma store ready at: {CHROMA_PERSIST_DIR.resolve()}")


if __name__ == "__main__":
    build_index()
