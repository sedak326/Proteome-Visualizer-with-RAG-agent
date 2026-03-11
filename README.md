# PLVis, Marine Mammals & RAG Agent

Proteomics data carries a lot of information about population health, ecological roles, and evolutionary history of species, but it is effectively inaccessible to anyone outside a narrow research community. Even specialists in adjacent fields often lack the training to interpret these findings, and the gap between what the data shows and what a non-specialist can actually understand is large.

To close that gap, we built PLVis, an interactive portal that visualizes the proteomes of five Southern California marine mammal species and integrates an AI agent that guides users through what they are seeing. The agent is always present, aware of the current screen state, and able to explain the visualization, answer biology questions grounded in scientific literature, and suggest what to explore next.

This project was developed in the Jinich Lab at UCSD.

![Demo](prerendered_animations/all_animals.gif)

---

## How to Use It

**1. Install dependencies**
```bash
pip install -r requirements_dash.txt
```

**2. Set your OpenAI API key**
```bash
export OPENAI_API_KEY=your_key_here
```

**3. Run the app**
```bash
python proteome_explorer_ux2.py
# Open http://localhost:8050
```

The ChromaDB vector store (`Marine_RAG/chroma_db/`) is included so the agent works out of the box. To rebuild the index from your own literature, place source documents in `Marine_RAG/data/` and run:
```bash
python Marine_RAG/build_index.py
```

Once the app is running, select one or more species from the top bar to load their proteins into the UMAP. Hover over any point for protein details. Draw a lasso to select a group of proteins, then ask the agent anything about what you see.

---

## Species

| Common name | Scientific name | Proteins |
|---|---|---|
| California sea lion | *Zalophus californianus* | 86,604 |
| Bottlenose dolphin | *Tursiops truncatus* | 82,376 |
| Gray whale | *Eschrichtius robustus* | 78,421 |
| Killer whale | *Orcinus orca* | 112,335 |
| Harbor seal | *Phoca vitulina* | 83,393 |

---

## The AI Agent

The agent is the primary way users interact with the portal. Every message is first passed to a router that classifies it and decides which tool to call.

**Conversational** ("Hi", "Thanks", "That's interesting"), the RAG pipeline is skipped and the model replies directly.

**Visual** ("What am I looking at?", "What does this cluster mean?", "Explain the top-left region"), no retrieval is performed. Instead the agent assembles a real-time description of the current screen state and uses that as context. This visual context has up to three components:
- A spatial overview: the UMAP canvas divided into a 3x3 grid with protein counts per species per region and any areas of cross-species overlap
- A cluster summary: the top 30 clusters ranked by size, each labelled as unique to one species or shared across multiple, with representative protein names
- A selection summary (only when the user has lassoed proteins): the top 10 Pfam/HMM domains across the selection, a cluster breakdown, and 5 representative protein entries with UniProt IDs and UMAP coordinates

This means the agent always knows exactly what the user is currently looking at, without any manual input from the user.

**Biology** ("Why do dolphins and orcas share so many clusters?", "What does this domain do?", "How do killer whale proteomes compare to harbor seals?"), the full RAG pipeline runs. The agent retrieves the most relevant chunks from the curated knowledge base, generates a grounded answer, and then reformulates it into plain language. Follow-up suggestions are appended automatically.

If retrieved chunks are not sufficiently relevant, the model falls back on general marine biology knowledge and flags the response as not sourced from the indexed literature, so the system degrades gracefully rather than returning confidently wrong answers.

---

## UMAP Visualization

Each protein was embedded using ProtT5-XL, a transformer-based protein language model pretrained on UniRef50, which produces fixed-length vectors capturing structural, functional, and evolutionary properties. These high-dimensional embeddings were reduced to 2D using UMAP.

All five species were projected into a single shared UMAP so that spatial relationships between species are meaningful: overlap in the plot reflects real similarity in the underlying embedding space. K-means clustering was then applied to the combined dataset to identify protein groups that are unique to one species or shared across multiple.

Each protein was annotated using SeqHub, which assigns Pfam/HMM domain labels and functional descriptions. These annotations power the hover labels, cluster descriptions, and the agent's visual context assembly.

---

## RAG Knowledge Base

The agent's biology knowledge comes from a curated corpus of scientific papers and books on marine mammal biology, taxonomy, proteomics, and the specific species in the portal.

Documents were segmented into 2,628 chunks (2,499 text, 32 tables, 97 image summaries). Images were described using GPT-4o mini. Each chunk carries structured metadata: document name, chapter, section title, and page number. Chunks were embedded with OpenAI's `text-embedding-3-small` (1,536 dimensions) and stored in ChromaDB. At query time the top 8 most similar chunks are retrieved by cosine similarity and passed to the model as context.

---

## Evaluation

The agent was evaluated against GPT-4o mini as a base model using 200 QA pairs generated with AutoRAG from the knowledge base corpus.

**Retrieval**

| Metric | Score |
|---|---|
| Recall@1 | 0.42 |
| Recall@8 | 0.855 |
| MRR | 0.60 |

The correct chunk was present in the top 8 results for 85.5% of queries.

**Generation**

| Metric | RAG | Base (GPT-4o mini) |
|---|---|---|
| SemScore | 0.818 | 0.714 |
| BERTScore | 0.926 | 0.895 |
| SemScore Win Rate | 81.5% | , |
| BERTScore Win Rate | 73% | , |
| Faithfulness | 0.866 | , |
| Latency (s) | 3.43 | 2.19 |

The RAG agent outperformed the base model on 81.5% of questions by SemScore and 73% by BERTScore. Faithfulness of 0.866 indicates that the large majority of responses stayed grounded in the retrieved context.

---

## Stack

Dash + Plotly (web UI and scatter plot), ChromaDB (vector database), LangChain (embedding and retrieval), OpenAI API (`text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation and routing), ProtT5-XL (protein sequence embeddings, computed offline), SeqHub (protein annotation)

---

## What's Next

The agent already meaningfully outperforms the base model on answer quality and factual grounding. The main area to improve is retrieval precision: Recall@1 of 0.42 means the single most relevant chunk often is not ranked first, even when it is present in the top 8. Next steps focus on enhancing agent performance through better retrieval strategies, reranking, and further prompt refinement to push faithfulness and generation quality higher.
