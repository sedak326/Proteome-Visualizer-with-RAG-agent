#!/usr/bin/env python3
"""
add_bert_score.py — Add BERTScore to existing eval results.

Reads eval/results_plvis.parquet (already has rag/base answers) and adds:
    rag_bert_score   — BERTScore F1 for RAG answer vs generation_gt
    base_bert_score  — BERTScore F1 for base answer vs generation_gt

Updates eval/results_plvis.parquet in place and refreshes eval/summary_plvis.csv.

Usage:
    cd PLVis-Extension-with-Marine-Mammals-and-RAG/Marine_RAG
    python add_bert_score.py
"""

from pathlib import Path
import pandas as pd
from bert_score import score as bert_score_fn

EVAL_DIR     = Path(__file__).resolve().parent / "eval"
RESULTS_PATH = EVAL_DIR / "results_plvis.parquet"
SUMMARY_PATH = EVAL_DIR / "summary_plvis.csv"

df = pd.read_parquet(RESULTS_PATH)
print(f"Loaded {len(df)} rows from {RESULTS_PATH}")

# BERTScore expects flat lists; use the first generation_gt as reference
references = [str(gt[0]) if hasattr(gt, '__len__') and len(gt) > 0 else str(gt) for gt in df["generation_gt"]]

print("Scoring RAG answers …")
_, _, rag_f1 = bert_score_fn(
    df["rag_answer"].tolist(), references, lang="en", verbose=True
)

print("Scoring base answers …")
_, _, base_f1 = bert_score_fn(
    df["base_answer"].tolist(), references, lang="en", verbose=True
)

df["rag_bert_score"]  = rag_f1.numpy()
df["base_bert_score"] = base_f1.numpy()
df["rag_bert_wins"]   = (df["rag_bert_score"] > df["base_bert_score"]).astype(int)

df.to_parquet(RESULTS_PATH, index=False)
print(f"\nUpdated {RESULTS_PATH}")

# Refresh summary
summary = pd.read_csv(SUMMARY_PATH)
summary["rag_bert_score_mean"]  = df["rag_bert_score"].mean()
summary["base_bert_score_mean"] = df["base_bert_score"].mean()
summary["rag_bert_win_rate"]    = df["rag_bert_wins"].mean()
summary["rag_bert_delta_mean"]  = (df["rag_bert_score"] - df["base_bert_score"]).mean()
summary.to_csv(SUMMARY_PATH, index=False)

print("\n── BERTScore Summary ────────────────────────────────")
print(f"  RAG  bert_score mean : {summary['rag_bert_score_mean'].values[0]:.4f}")
print(f"  Base bert_score mean : {summary['base_bert_score_mean'].values[0]:.4f}")
print(f"  RAG delta (mean)     : {summary['rag_bert_delta_mean'].values[0]:+.4f}")
print(f"  RAG bert win rate    : {summary['rag_bert_win_rate'].values[0]:.1%}")
print(f"\nUpdated {SUMMARY_PATH}")
