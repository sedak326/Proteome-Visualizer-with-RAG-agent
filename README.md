# PLVis — Protein Language Visualizer

PLVis is an interactive web application for exploring and comparing marine mammal proteomes. It visualizes tens of thousands of proteins in a shared 2D space using UMAP, with each dot representing a single protein colored by species. A built-in AI assistant, powered by a RAG pipeline grounded in curated scientific literature, lets users ask questions about what they're seeing, from "what do these proteins have in common?" to "why do dolphins and orcas share so many clusters?"

The project was developed in the Jinich Lab at UCSD as part of ongoing work on proteome visualization and conservation-oriented science communication.

![Demo](prerendered_animations/all_animals.gif)

---

## Motivation

Proteomes encode a surprising amount of information about the biology of an organism, population health, ecological roles, evolutionary history, and more. At the scale of an entire proteome, clustering approaches can reveal both deep conservation across species and lineage-specific novelty. This kind of analysis has clear relevance to marine conservation: species like the harbor seal and bottlenose dolphin are umbrella species in Southern California, and understanding their biology at the molecular level can inform how we protect them.

The problem is that this data is effectively inaccessible to anyone outside a narrow research community. A comprehensive literature review found that most people have only a very limited understanding of omics data, and this isn't just a public education problem. Even specialists in adjacent fields often aren't trained to interpret proteomics findings. The gap between what the data shows and what anyone outside the field can actually read and use is large.

PLVis is an attempt to close that gap. Rather than treating visualization and explanation as separate steps, it couples them: you can load two species, look at where their proteins overlap in the UMAP, lasso a cluster that catches your attention, and immediately ask what those proteins do and why the species might share them. The AI assistant grounds its answers in real literature, so responses are accurate rather than hallucinated, and written in plain language rather than jargon.

---

## Species

Five marine mammal species from Southern California, plus one terrestrial outgroup included to make cross-species clustering patterns visually obvious:

| Common name | Scientific name | Source |
|---|---|---|
| Gray whale | *Eschrichtius robustus* | NCBI |
| Orca | *Orcinus orca* | NCBI |
| California sea lion | *Zalophus californianus* | UniProt |
| Harbor seal | *Phoca vitulina* | NCBI |
| Bottlenose dolphin | *Tursiops truncatus* | UniProt |
| King cobra | *Ophiophagus hannah* | — |

---

## UMAP Visualization

Protein sequences can't be compared or visualized directly, they need to be converted into a numerical representation that captures their biological properties. Each protein in PLVis was embedded using **ProtT5-XL**, a transformer-based protein language model pretrained on UniRef50. Like language models for text, ProtT5-XL learns the "grammar" of amino acid sequences, producing high-dimensional embeddings where proteins with similar structure, function, or evolutionary origin end up geometrically close to each other in vector space.

These embeddings are high-dimensional by nature (each protein is represented as a long numerical vector), so visualizing them requires dimensionality reduction. PLVis uses **UMAP**, which works in two phases: first constructing a graph of nearest neighbors in the high-dimensional space, then optimizing a low-dimensional layout that preserves those neighbor relationships. Compared to alternatives like t-SNE, UMAP better preserves global structure and scales to the dataset sizes involved here.

Critically, all species were embedded into a **single shared UMAP**. Computing species separately and then overlaying them would produce incompatible coordinate spaces. By projecting all proteins together and then splitting by species, the spatial relationships between species are meaningful, overlap in the plot reflects real similarity in the underlying embedding space.

After computing UMAP coordinates, **K-means clustering** was applied to the combined dataset. Clusters that appear in multiple species are labelled *shared*; those found in only one species are labelled *unique*. This is one of the more informative things to explore in the interface.

Each protein was also annotated using **SeqHub** (formerly Gaia), which matches query sequences to a curated reference database and assigns Pfam/HMM domain labels, functional descriptions, and taxonomic information. These annotations are what power the hover labels, cluster descriptions, and lasso summaries in the app.

### Lasso selection

When you draw a lasso selection on the plot, the app runs a single pass over all selected points and computes three things: the **top 10 most frequent Pfam/HMM domains** across the selection (the fastest signal for "what do these proteins have in common?"), a **cluster breakdown** showing the 10 largest clusters each labelled as unique to a species or shared across multiple, and **five representative sample proteins** with UniProt IDs, descriptions, UMAP coordinates, and cluster assignments.

The prompt size stays roughly constant regardless of how many proteins you select, so a lasso of 500 proteins and a lasso of 20 produce comparably sized context strings for the assistant.

---

## RAG Assistant

The assistant was designed to handle three types of questions that users are likely to ask: how to read and interpret the UMAP, questions about the taxonomic and evolutionary relationships between the featured species, and questions about specific proteins or protein families. The knowledge base was assembled accordingly.

### Literature

| Document | Type | Coverage |
|---|---|---|
| | | |
| | | |
| | | |

*(Add your sources here)*

### Indexing

The literature was segmented into chunks before indexing:

**HTML documents** are split at headings (e.g. "III Ecology", "V Life History"). A 4,000-character limit per chunk is enforced; sections that exceed it are further subdivided. Large chunks hurt retrieval because they contain too much peripheral content relative to what any given query is asking, smaller chunks keep retrieval focused.

**PDF documents** use the same 4,000-character limit. Since PDFs have no structural tags, section boundaries are identified heuristically: any bold text on a page is treated as a heading and starts a new chunk. Repeated short text blocks (running headers/footers appearing more than three times and under 200 characters) are filtered out. Figure and table captions are skipped. Processing stops at terminal sections like References, Acknowledgements, or Supplementary Material. Images are extracted and described via `gpt-4o-mini`; those descriptions are indexed as text chunks. Tables are extracted as TSV.

Total: **2,628 chunks** , 2,499 text, 32 tables, 97 image summaries. Each chunk carries metadata: document name, chapter, section title, page number.

### Retrieval

Chunks were embedded using OpenAI's `text-embedding-3-small` model, producing 1,536-dimensional vectors stored in **ChromaDB**. At query time, the user's question is embedded with the same model and ChromaDB performs a cosine similarity search, returning the 8 most relevant chunks. Using the same model at both indexing and retrieval time is important, different models produce incompatible vector spaces that would make similarity comparisons meaningless.

Retrieved chunks, with their metadata, are passed to `gpt-4o-mini` as context. If the retrieved chunks aren't sufficiently relevant, the model falls back on its general marine biology knowledge and explicitly flags the response as not sourced from the indexed literature, so the system degrades gracefully rather than confidently returning something wrong. The raw answer then goes through a second LLM call that rewrites it in plain, accessible language and appends follow-up topic suggestions.

### Message routing

Every message passes through a router LLM before anything else. The router classifies the message into one of three categories: **Conversational** ("Hi!", "Cool!", "Thanks"), where the RAG pipeline is skipped entirely and `gpt-4o-mini` replies directly, **Visual** ("What do I see?", "Explain this cluster", "What's in the top-left?"), where no retrieval is performed and a pre-assembled description of the current screen state is passed as context instead, and **Biology** (questions about specific proteins, species, domains, or anything that needs domain knowledge), where the full RAG pipeline runs.

The visual context description has two parts. A **spatial overview** is always present: the UMAP canvas is divided into a 3×3 grid and the app counts how many proteins from each species fall in each region, flagging any regions where species overlap. The LLM receives this as readable text rather than raw coordinates. A **selection summary** is added only when the user has made a lasso selection, using the domain, cluster, and sample data described above.

---

## Stack

**Dash + Plotly** (web UI and interactive scatter plot), **ChromaDB** (vector database), **LangChain** (embedding and retrieval wrappers), **OpenAI API** (`text-embedding-3-small` for embeddings, `gpt-4o-mini` for generation and routing), **ProtT5-XL** (protein sequence embeddings, computed offline), **SeqHub** (protein annotation), **OpenCV** (animation processing)

---

## Setup

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
# → http://localhost:8050
```

The ChromaDB vector store (`Marine_RAG/chroma_db/`) is included in the repo, so the RAG assistant works out of the box. If you want to rebuild the index from your own literature, place your source documents in `Marine_RAG/data/` and run:
```bash
python Marine_RAG/build_index.py
```

> **Note:** The proteome data files (`umap_output/`, `annotated/`) are included. The raw FASTA files and ProtT5 embeddings used to generate them are not — they're several GB and not needed to run the app.
