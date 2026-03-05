"""
create_qa_plvis.py — Generate QA pairs from the PLVis RAG corpus.

Inputs:
    eval/raw_plvis.parquet    — one row per source document (built by build_eval_corpus.py)
    eval/corpus_plvis.parquet — one row per ChromaDB chunk  (built by build_eval_corpus.py)

Outputs:
    eval/qa_plvis.parquet     — QA pairs with retrieval_gt pointing to ChromaDB chunk UUIDs
    eval/corpus_plvis.parquet — rewritten by AutoRAG (same data, AutoRAG-normalised format)

Run from PLVis-Extension-with-Marine-Mammals-and-RAG/Marine_RAG/ with the autorag conda env:
    conda activate autorag
    python create_qa_plvis.py
"""

# ── Compatibility shim ────────────────────────────────────────────────────────
# langchain-core 0.3+ removed pydantic_v1; langchain-upstage 0.1.5 (pinned by
# autorag 0.3.21) still imports it.  Inject pydantic.v1 under the old name
# before any langchain/autorag module loads.
import sys
import pydantic.v1 as _pydantic_v1
sys.modules.setdefault("langchain_core.pydantic_v1", _pydantic_v1)
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
    make_concise_gen_gt,
)
from autorag.data.qa.schema import Raw, Corpus
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop

EVAL_DIR    = "eval"
N_QUESTIONS = 200  # larger than the AI_Agent sets since this is our primary eval corpus

llm = OpenAI(model="gpt-4o-mini")

raw_df    = pd.read_parquet(f"{EVAL_DIR}/raw_plvis.parquet")
corpus_df = pd.read_parquet(f"{EVAL_DIR}/corpus_plvis.parquet")

print(f"Raw documents : {len(raw_df)}")
print(f"Corpus chunks : {len(corpus_df)}")

raw_instance    = Raw(raw_df)
corpus_instance = Corpus(corpus_df, raw_instance)

qa = (
    corpus_instance.sample(random_single_hop, n=N_QUESTIONS)
    .map(lambda df: df.reset_index(drop=True))
    .make_retrieval_gt_contents()
    .batch_apply(factoid_query_gen, llm=llm)
    .batch_apply(make_basic_gen_gt, llm=llm)
    .batch_apply(make_concise_gen_gt, llm=llm)
    .filter(dontknow_filter_rule_based, lang="en")
)

qa.to_parquet(f"{EVAL_DIR}/qa_plvis.parquet", f"{EVAL_DIR}/corpus_plvis.parquet")
print(f"Done — QA pairs saved to {EVAL_DIR}/qa_plvis.parquet")
