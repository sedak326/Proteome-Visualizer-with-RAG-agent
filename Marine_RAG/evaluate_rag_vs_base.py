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

# AutoRAG faithfulness (run in autorag conda env)
try:
    from autorag.evaluation.metric.generation import deepeval_faithfulness
    from autorag.schema.metricinput import MetricInput
    AUTORAG_AVAILABLE = True
except ImportError:
    AUTORAG_AVAILABLE = False

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


def rag_retrieve(question: str) -> tuple[str, list[str], list[str]]:
    """Returns (answer, retrieved_ids, retrieved_docs)."""
    vs = _load_vectorstore()
    q_embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
    result = vs._collection.query(
        query_embeddings=[q_embedding],
        n_results=RAG_TOP_K,
        include=["documents"],
    )
    retrieved_ids = result["ids"][0] if result["ids"] else []
    docs = result["documents"][0] if result["documents"] else []
    context = "\n\n---\n\n".join(docs) if docs else ""
    # Strip null bytes and control characters that break JSON serialisation
    context = "".join(c for c in context if c >= " " or c in "\n\t")

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
    return resp.choices[0].message.content.strip(), retrieved_ids, docs


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], gt_ids: list[str], k: int) -> float:
    """1.0 if any of the top-k retrieved IDs appear in gt_ids, else 0.0."""
    return float(bool(set(retrieved_ids[:k]) & set(gt_ids)))


def mrr(retrieved_ids: list[str], gt_ids: list[str]) -> float:
    """Reciprocal rank of the first retrieved ID that appears in gt_ids."""
    gt_set = set(gt_ids)
    for rank, uid in enumerate(retrieved_ids, start=1):
        if uid in gt_set:
            return 1.0 / rank
    return 0.0


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
        gen_gt   = list(row["generation_gt"])  # generation_gt consists of a long-form + concise version of the answer

        print(f"[{i+1}/{len(qa_df)}] {question[:80]}", end=" … ", flush=True)

        #retrieval_gt is the ID of the correct chunk
        retrieval_gt = list(row["retrieval_gt"][0]) if hasattr(row["retrieval_gt"][0], "__iter__") else list(row["retrieval_gt"])
        
        #calculates how long it takes for RAG agent and base model to generate answer
        t0 = time.time()
        rag_ans, retrieved_ids, retrieved_docs = rag_retrieve(question)
        rag_latency = time.time() - t0

        t0 = time.time()
        base_ans = base_answer(question)
        base_latency = time.time() - t0

        rag_sem  = sem_score(rag_ans,  gen_gt)
        base_sem = sem_score(base_ans, gen_gt)
        grounded = is_grounded(rag_ans)

        r_at_1 = recall_at_k(retrieved_ids, retrieval_gt, k=1)
        r_at_8 = recall_at_k(retrieved_ids, retrieval_gt, k=RAG_TOP_K)
        mrr_score = mrr(retrieved_ids, retrieval_gt)

        print(f"rag={rag_sem:.3f}  base={base_sem:.3f}  grounded={grounded}  R@1={r_at_1:.0f}  R@8={r_at_8:.0f}  MRR={mrr_score:.3f}  rag_t={rag_latency:.2f}s  base_t={base_latency:.2f}s")

        records.append({
            "qid":            row["qid"],
            "query":          question,
            "rag_answer":     rag_ans,
            "base_answer":    base_ans,
            "generation_gt":  gen_gt,
            "retrieval_gt":   retrieval_gt,
            "retrieved_ids":  retrieved_ids,
            "rag_sem_score":  rag_sem,
            "base_sem_score": base_sem,
            "rag_grounded":   grounded,
            "rag_wins":       int(rag_sem > base_sem),
            "recall_at_1":    r_at_1,
            "recall_at_8":    r_at_8,
            "mrr":            mrr_score,
            "rag_latency_s":  rag_latency,
            "base_latency_s": base_latency,
            "retrieved_docs": retrieved_docs,
        })

        # Avoid hitting rate limits
        time.sleep(0.5)

    results_df = pd.DataFrame(records)

    # Faithfulness scoring via AutoRAG deepeval_faithfulness
    if AUTORAG_AVAILABLE:
        print("\nScoring faithfulness via AutoRAG deepeval_faithfulness …")
        metric_inputs = [
            MetricInput(
                generated_texts=row["rag_answer"],
                retrieval_gt_contents=[[doc] for doc in row["retrieved_docs"]],
            )
            for _, row in results_df.iterrows()
        ]
        faithfulness_scores = deepeval_faithfulness(
            metric_inputs, generator_module_type="openai_llm", llm="gpt-4o-mini"
        )
        results_df["faithfulness"] = faithfulness_scores
    else:
        print("\nAutoRAG not available — skipping faithfulness scoring.")
        results_df["faithfulness"] = None

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
        "recall_at_1_mean":    results_df["recall_at_1"].mean(),
        "recall_at_8_mean":    results_df["recall_at_8"].mean(),
        "mrr_mean":            results_df["mrr"].mean(),
        "rag_latency_mean_s":  results_df["rag_latency_s"].mean(),
        "base_latency_mean_s": results_df["base_latency_s"].mean(),
        "latency_overhead_s":  (results_df["rag_latency_s"] - results_df["base_latency_s"]).mean(),
        "faithfulness_mean":   results_df["faithfulness"].mean() if AUTORAG_AVAILABLE else None,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\n── Generation ───────────────────────────────────")
    print(f"  Questions evaluated : {n}")
    print(f"  RAG  sem_score mean : {summary['rag_sem_score_mean']:.4f}")
    print(f"  Base sem_score mean : {summary['base_sem_score_mean']:.4f}")
    print(f"  RAG delta (mean)    : {summary['rag_delta_mean']:+.4f}")
    print(f"  RAG win rate        : {summary['rag_win_rate']:.1%}")
    print(f"  RAG grounded rate   : {summary['rag_grounded_rate']:.1%}")
    print("\n── Retrieval ────────────────────────────────────")
    print(f"  Recall@1            : {summary['recall_at_1_mean']:.4f}")
    print(f"  Recall@8            : {summary['recall_at_8_mean']:.4f}")
    print(f"  MRR                 : {summary['mrr_mean']:.4f}")
    if AUTORAG_AVAILABLE:
        print("\n── Faithfulness ─────────────────────────────────")
        print(f"  Faithfulness mean   : {summary['faithfulness_mean']:.4f}")
    print("\n── Latency ──────────────────────────────────────")
    print(f"  RAG  mean latency   : {summary['rag_latency_mean_s']:.3f}s")
    print(f"  Base mean latency   : {summary['base_latency_mean_s']:.3f}s")
    print(f"  RAG overhead (mean) : {summary['latency_overhead_s']:+.3f}s")
    print(f"\nSaved summary → {SUMMARY_PATH}")


if __name__ == "__main__":
    run_evaluation()
