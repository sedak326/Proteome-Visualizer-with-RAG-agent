#!/usr/bin/env python3
"""
evaluate_rag_vs_base.py — Compare the PLVis RAG pipeline against a bare GPT-4o-mini baseline.

For each question in eval/qa_plvis.parquet the script:
  1. Calls the RAG pipeline  (ChromaDB retrieval + gpt-4o-mini)
  2. Calls the base model    (gpt-4o-mini, no retrieval context)
  3. Scores both answers against generation_gt using:
       sem_score   — cosine similarity in OpenAI text-embedding-3-small space
       grounded    — 1 if the RAG answer cites the corpus, 0 if it fell back to
                     general knowledge (detected via the explicit fallback phrase
                     written into query_rag's system prompt)

Outputs:
  eval/results_plvis.parquet  — per-question scores for both systems
  eval/summary_plvis.csv      — mean scores and win-rate summary

Usage:
    cd PLVis-Extension-with-Marine-Mammals-and-RAG/Marine_RAG
    python evaluate_rag_vs_base.py

Requires:
    OPENAI_API_KEY set in environment
    pip install langchain-chroma langchain-openai openai pandas numpy
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_DIR           = Path(__file__).resolve().parent / "eval"
QA_PATH            = EVAL_DIR / "qa_plvis.parquet"
RESULTS_PATH       = EVAL_DIR / "results_plvis.parquet"
SUMMARY_PATH       = EVAL_DIR / "summary_plvis.csv"

CHROMA_PERSIST_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME    = "marine_rag_v1"
EMBEDDING_MODEL    = "text-embedding-3-small"
RAG_TOP_K          = 8

FALLBACK_PHRASE    = "This is from general knowledge, not the specific literature."

client     = OpenAI()
_vectorstore: Chroma | None = None


# ---------------------------------------------------------------------------
# RAG pipeline  (mirrors query_rag() in proteome_explorer_ux2.py)
# ---------------------------------------------------------------------------

def _load_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    return _vectorstore


def rag_answer(question: str) -> str:
    vs = _load_vectorstore()
    retrieved = vs.similarity_search(question, k=RAG_TOP_K)
    context = "\n\n---\n\n".join(d.page_content for d in retrieved) if retrieved else ""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a marine biology knowledge assistant. "
                "You have access to a curated corpus of marine mammal research papers "
                "and species accounts (context chunks below). "
                "Prioritise answering from the context — each chunk starts with its source "
                "location (Document, Chapter, Section, Page), so cite naturally "
                "(e.g. 'According to Smith 2023, page 4...'). "
                "If the context chunks are not relevant to the question, answer from your "
                "general marine biology knowledge but clearly note: "
                f"'{FALLBACK_PHRASE}' "
                "Never fabricate citations or paper titles."
            )},
            {"role": "user", "content": (
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                if context else
                f"Question: {question}\n\nAnswer:"
            )},
        ],
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Base model  (no retrieval, no system context about the corpus)
# ---------------------------------------------------------------------------

def base_answer(question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a marine biology expert. "
                "Answer the question concisely and accurately from your general knowledge."
            )},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts with text-embedding-3-small."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return np.array([d.embedding for d in response.data])


def sem_score(answer: str, references: list[str]) -> float:
    """
    Cosine similarity between the answer embedding and the mean of reference
    embeddings.  references is generation_gt (list of acceptable answers).
    """
    texts = [answer] + references
    embs = _embed(texts)
    a = embs[0]
    r = embs[1:].mean(axis=0)
    return float(np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r) + 1e-9))


def is_grounded(answer: str) -> int:
    """1 if the RAG answered from the corpus, 0 if it fell back to general knowledge."""
    return 0 if FALLBACK_PHRASE in answer else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    qa_df = pd.read_parquet(QA_PATH)
    print(f"Loaded {len(qa_df)} questions from {QA_PATH}")
    print("Running RAG and base model — this will take a few minutes …\n")

    records = []
    for i, row in qa_df.iterrows():
        question = row["query"]
        gen_gt   = list(row["generation_gt"])  # list of acceptable answers

        print(f"[{i+1}/{len(qa_df)}] {question[:80]}", end=" … ", flush=True)

        rag_ans  = rag_answer(question)
        base_ans = base_answer(question)

        rag_sem  = sem_score(rag_ans,  gen_gt)
        base_sem = sem_score(base_ans, gen_gt)
        grounded = is_grounded(rag_ans)

        print(f"rag={rag_sem:.3f}  base={base_sem:.3f}  grounded={grounded}")

        records.append({
            "qid":           row["qid"],
            "query":         question,
            "rag_answer":    rag_ans,
            "base_answer":   base_ans,
            "generation_gt": gen_gt,
            "rag_sem_score": rag_sem,
            "base_sem_score":base_sem,
            "rag_grounded":  grounded,
            "rag_wins":      int(rag_sem > base_sem),
        })

        # Avoid hitting rate limits
        time.sleep(0.5)

    results_df = pd.DataFrame(records)
    results_df.to_parquet(RESULTS_PATH, index=False)
    print(f"\nSaved per-question results → {RESULTS_PATH}")

    # Summary
    n = len(results_df)
    summary = {
        "n_questions":        n,
        "rag_sem_score_mean": results_df["rag_sem_score"].mean(),
        "base_sem_score_mean":results_df["base_sem_score"].mean(),
        "rag_win_rate":       results_df["rag_wins"].mean(),
        "rag_grounded_rate":  results_df["rag_grounded"].mean(),
        "rag_delta_mean":     (results_df["rag_sem_score"] - results_df["base_sem_score"]).mean(),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Questions evaluated : {n}")
    print(f"  RAG  sem_score mean : {summary['rag_sem_score_mean']:.4f}")
    print(f"  Base sem_score mean : {summary['base_sem_score_mean']:.4f}")
    print(f"  RAG delta (mean)    : {summary['rag_delta_mean']:+.4f}")
    print(f"  RAG win rate        : {summary['rag_win_rate']:.1%}")
    print(f"  RAG grounded rate   : {summary['rag_grounded_rate']:.1%}")
    print(f"\nSaved summary → {SUMMARY_PATH}")


if __name__ == "__main__":
    run_evaluation()
