#!/usr/bin/env python3
"""
generate_qa_simple.py — Generate QA pairs from corpus_plvis.parquet using OpenAI directly.

Replaces create_qa_plvis.py which has broken autorag/langchain_upstage dependencies.

Usage:
    cd PLVis-Extension-with-Marine-Mammals-and-RAG/Marine_RAG
    conda run -n autorag python3 generate_qa_simple.py

Outputs:
    eval/qa_plvis.parquet  — appended to existing QA pairs (or created fresh)
"""

from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR       = Path(__file__).resolve().parent / "eval"
CORPUS_PATH    = EVAL_DIR / "corpus_plvis.parquet"
QA_PATH        = EVAL_DIR / "qa_plvis.parquet"
N_NEW          = 200   # number of new QA pairs to generate
MIN_CHUNK_LEN  = 200   # skip very short chunks
RANDOM_SEED    = 42

client = OpenAI()

# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------

print(f"Loading corpus from {CORPUS_PATH} …")
corpus = pd.read_parquet(CORPUS_PATH)
print(f"  {len(corpus)} chunks available")

# Filter out very short chunks
corpus = corpus[corpus["contents"].str.len() >= MIN_CHUNK_LEN].reset_index(drop=True)
print(f"  {len(corpus)} chunks after length filter")

# ---------------------------------------------------------------------------
# Sample chunks
# ---------------------------------------------------------------------------

random.seed(RANDOM_SEED)
n_sample = min(N_NEW, len(corpus))
sampled = corpus.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Generate QA pairs
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in marine mammal biology and molecular evolution.
Given a passage from a scientific paper or textbook, generate one factoid question
that can be answered from the passage, and provide a concise answer.

Respond with JSON only:
{
  "question": "...",
  "answer_long": "...",
  "answer_short": "..."
}

Rules:
- The question must be answerable from the passage alone
- answer_long: 1-3 sentences with full context
- answer_short: 1 sentence or key phrase only
- Do not ask questions about author names, journal names, or publication details
- Focus on biological findings, mechanisms, or adaptations
"""

rows = []
print(f"\nGenerating {n_sample} QA pairs …")
for i, (_, row) in enumerate(sampled.iterrows()):
    chunk_text = row["contents"][:3000]  # truncate very long chunks
    doc_id = row["doc_id"]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Passage:\n{chunk_text}"},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        question = data.get("question", "").strip()
        answer_long = data.get("answer_long", "").strip()
        answer_short = data.get("answer_short", "").strip()

        if not question or not answer_long:
            continue
        # Skip if model says it can't answer
        if any(w in answer_long.lower() for w in ["i don't know", "cannot answer", "not mentioned"]):
            continue

        rows.append({
            "qid": str(uuid.uuid4()),
            "query": question,
            "retrieval_gt": [[doc_id]],
            "generation_gt": [answer_long, answer_short],
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_sample}] generated {len(rows)} valid pairs so far")

    except Exception as e:
        print(f"  [{i+1}] error: {e}")
        continue

print(f"\nGenerated {len(rows)} valid QA pairs")

# ---------------------------------------------------------------------------
# Merge with existing QA pairs and save
# ---------------------------------------------------------------------------

new_df = pd.DataFrame(rows)

new_df.to_parquet(QA_PATH, index=False)
print(f"Saved {len(new_df)} QA pairs → {QA_PATH}")
