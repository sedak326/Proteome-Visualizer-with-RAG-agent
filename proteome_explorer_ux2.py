"""
Marine Mammal Proteome Explorer - Ocean Theme Edition
User Flow: Select Animals (with icons) → Watch Animation → Explore Proteomes
Deep sea ocean theme with bioluminescent accents
"""

import re
import json
import dash
from scipy.spatial import ConvexHull
from scipy.stats import hypergeom
from dash import dcc, html, Input, Output, State, ctx, dash_table, ALL
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import base64
import subprocess
import sys
from openai import OpenAI as OpenAIClient
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Resolve all paths relative to this script, not the working directory
BASE_DIR = Path(__file__).resolve().parent

# RAG Chat Configuration
CHROMA_PERSIST_DIR = str(BASE_DIR / "Marine_RAG" / "chroma_db")
CHROMA_COLLECTION  = "marine_rag_v1"
RAG_TOP_K          = 8
_openai_client     = OpenAIClient()
_vectorstore       = None


def _load_vectorstore() -> Chroma:
    """Load (or return cached) Chroma vector store."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    _vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )
    count = _vectorstore._collection.count()
    print(f"Chroma vectorstore loaded — {count} chunks indexed")
    return _vectorstore


def _region_label(x, y):
    """Map UMAP coordinates to a human-readable region name."""
    row = "top" if y > 100 else ("bottom" if y < -100 else "middle")
    col = "left" if x < -100 else ("right" if x > 100 else "center")
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def build_spatial_summary(active_species, loaded_data):
    """Create a compact spatial description of where each species' proteins sit on the UMAP plot."""
    if not loaded_data or not active_species:
        return ""
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    if not active_data:
        return ""

    # Count proteins per (species, region)
    region_species = {}   # region -> {species: count}
    species_totals = {}
    for sd in active_data:
        name = sd['name']
        xs = sd.get('umap_1_scaled', [])
        ys = sd.get('umap_2_scaled', [])
        species_totals[name] = len(xs)
        for x, y in zip(xs, ys):
            reg = _region_label(x, y)
            region_species.setdefault(reg, {})
            region_species[reg][name] = region_species[reg].get(name, 0) + 1

    lines = ["Spatial layout of the UMAP plot (each region ~1/9 of the area):"]
    for reg in ["top-left", "top", "top-right", "left", "center", "right",
                "bottom-left", "bottom", "bottom-right"]:
        counts = region_species.get(reg, {})
        if not counts:
            continue
        parts_r = []
        for sp, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = round(100 * cnt / species_totals[sp]) if species_totals.get(sp) else 0
            parts_r.append(f"{sp} {cnt} ({pct}%)")
        overlap_note = ""
        if len(counts) > 1:
            overlap_note = " [OVERLAP]"
        lines.append(f"  {reg}: {', '.join(parts_r)}{overlap_note}")

    return "\n".join(lines)


def build_cluster_summary(active_species, loaded_data):
    """Build a per-cluster breakdown from loaded_data (always available, no selection needed)."""
    if not loaded_data or not active_species:
        return ""
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    if not active_data:
        return ""

    # cluster_id -> {species_name -> [protein_names]}
    cluster_info = {}
    for sd in active_data:
        species_name = sd['name']
        display = SPECIES_DATA.get(species_name, {}).get('display_name', species_name)
        for rec in sd.get('df', []):
            cl = rec.get('Cluster Label')
            if cl is None or str(cl) == 'nan':
                continue
            cl = int(cl)
            if cl < 0:
                continue
            cluster_info.setdefault(cl, {})
            cluster_info[cl].setdefault(display, [])
            pname = str(rec.get('Protein names', '') or '').split('(')[0].strip()[:60]
            if not pname:
                entry_id = str(rec.get('Entry', '') or '')
                ann = ANNOTATIONS.get(entry_id, {})
                pname = (ann.get('desc', '') or '').split(' OS=')[0].strip()[:60]
            if pname:
                cluster_info[cl][display].append(pname)

    if not cluster_info:
        return ""

    # Determine which clusters are unique vs shared
    lines = ["Cluster summary (all clusters visible on screen):"]
    sorted_clusters = sorted(cluster_info.items(), key=lambda x: -sum(len(v) for v in x[1].values()))
    for cl, species_proteins in sorted_clusters[:30]:  # cap at 30 clusters
        all_species = list(species_proteins.keys())
        sharing = "unique to " + all_species[0] if len(all_species) == 1 else "shared (" + ", ".join(all_species) + ")"
        total = sum(len(v) for v in species_proteins.values())
        # Sample up to 3 representative protein names
        sample_names = []
        for names_list in species_proteins.values():
            for n in names_list:
                if n not in sample_names:
                    sample_names.append(n)
                if len(sample_names) >= 3:
                    break
            if len(sample_names) >= 3:
                break
        sample_str = "; ".join(sample_names) if sample_names else "no name info"
        lines.append(f"  Cluster {cl} ({sharing}): {total} proteins — e.g. {sample_str}")

    return "\n".join(lines)


def build_visual_context(active_species, selected_data, loaded_data=None):
    """Build a text summary of what's currently on screen."""
    parts = []
    active_species = active_species or []

    if active_species:
        names = [SPECIES_DATA[s]['display_name'] for s in active_species if s in SPECIES_DATA]
        parts.append(f"Species displayed: {', '.join(names)}")

    spatial = build_spatial_summary(active_species, loaded_data)
    if spatial:
        parts.append(spatial)

    cluster_summary = build_cluster_summary(active_species, loaded_data)
    if cluster_summary:
        parts.append(cluster_summary)

    if selected_data and selected_data.get('points'):
        # Build flat entry_id -> species lookup (avoids curveNumber which is offset by hull traces)
        entry_to_species = {}
        if loaded_data:
            for sd in loaded_data:
                if sd['name'] in active_species:
                    for rec in sd['df']:
                        entry_to_species[str(rec.get('Entry', ''))] = sd['name']

        # Single pass — collect everything needed for aggregation and sampling
        cluster_species = {}   # cluster_label -> set of species keys
        species_counts  = {}   # species key -> int
        domain_counts   = {}   # domain label -> int (across all proteins)
        cluster_counts  = {}   # cluster_label -> int
        sample_points   = []   # list of formatted lines for the representative sample

        for point in selected_data['points']:
            customdata = point.get('customdata', [])
            if not customdata or len(customdata) < 2:
                continue  # skip hull/centroid traces (customdata length < 2)
            entry_id = str(customdata[0])
            species = entry_to_species.get(entry_id)
            if not species:
                continue
            species_counts[species] = species_counts.get(species, 0) + 1

            hover = point.get('text', '')
            cluster_label = ''
            if 'Cluster:' in hover:
                cluster_label = hover.split('Cluster:')[-1].strip().split('<')[0].strip()
            if cluster_label:
                cluster_species.setdefault(cluster_label, set()).add(species)
                cluster_counts[cluster_label] = cluster_counts.get(cluster_label, 0) + 1

            ann = ANNOTATIONS.get(entry_id, {})

            # Aggregate domain frequencies across the full selection
            for dom in (ann.get('hmm_labels', '') or '').split(';'):
                dom = dom.strip()
                if dom:
                    domain_counts[dom] = domain_counts.get(dom, 0) + 1

            # Collect up to 5 sample proteins total (first encountered)
            if len(sample_points) < 5:
                desc     = ann.get('desc', '').split(' OS=')[0].strip() or 'Unknown'
                hmm_desc = ann.get('hmm_descriptions', '') or ''
                domains  = ann.get('hmm_labels', '') or 'None'
                x = round(point.get('x', 0), 1)
                y = round(point.get('y', 0), 1)
                sharing = (
                    "unique" if len(cluster_species.get(cluster_label, set())) == 1 else "shared"
                ) if cluster_label else ""
                line = (
                    f"  - {entry_id} ({SPECIES_DATA[species]['display_name']}): "
                    f"{desc[:80]}, UMAP=({x},{y})"
                )
                if cluster_label:
                    line += f", Cluster: {cluster_label} ({sharing})"
                line += f", Domains: {(domains[:60])}"
                if hmm_desc:
                    line += f", Function: {hmm_desc[:80]}"
                sample_points.append(line)

        total = len(selected_data['points'])
        breakdown = ", ".join(
            f"{SPECIES_DATA[sp]['display_name']}: {cnt}"
            for sp, cnt in species_counts.items()
        )

        section = [f"Selected proteins ({total} total — {breakdown}):"]

        # Domain frequency summary (top 10)
        if domain_counts:
            top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:10]
            section.append("  Top domains across selection:")
            for dom, cnt in top_domains:
                section.append(f"    {dom}: {cnt} proteins")

        # Cluster breakdown (top 10 by size, labelled unique/shared)
        if cluster_counts:
            top_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])[:10]
            section.append("  Cluster breakdown:")
            for cl, cnt in top_clusters:
                sp_set = cluster_species.get(cl, set())
                sharing = "unique to " + SPECIES_DATA[next(iter(sp_set))]['display_name'] \
                    if len(sp_set) == 1 else "shared"
                section.append(f"    Cluster {cl} ({sharing}): {cnt} proteins")

        # Representative sample
        if sample_points:
            section.append(f"  Representative sample ({len(sample_points)} of {total}):")
            section.extend(sample_points)

        parts.append("\n".join(section))

    return "\n".join(parts) if parts else ""


def route_message(chat_history, new_message, visual_context=""):
    """Classify the message and route to RAG, visual context, or conversational reply.

    Returns (mode, text):
        ("rag", standalone_query) - send to RAG pipeline
        ("visual", question)     - answer using current screen context
        ("chat", direct_reply)   - return directly, no retrieval needed
    """
    context = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in (chat_history or [])[-14:]
    )
    user_content = f"Conversation:\n{context}\n\n"
    if visual_context:
        user_content += f"Currently on screen:\n{visual_context}\n\n"
    user_content += f"Latest message: {new_message}"
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a router for an enthusiastic nerdy-scientist lab guide on a proteome "
                    "visualization portal. Given a conversation, what the user sees on screen, "
                    "and their latest message, decide:\n"
                    "- CONTROL: <command> — use when the user wants to change a plot setting: "
                    "filter proteins ('show unique proteins', 'show shared proteins', 'show all proteins'), "
                    "adjust point size ('make the dots bigger', 'set size to 7', 'smaller points'), "
                    "adjust opacity ('make it more transparent', 'set opacity to 0.5'), "
                    "or asks about zoom/pan/screenshot (toolbar actions). "
                    "Output 'CONTROL: ' followed by the command verbatim, nothing else.\n"
                    "- SPECIES: <action> <species name> — use when the user wants to add, remove, "
                    "or change which species are shown on the plot. Actions: 'add <name>', "
                    "'remove <name>', 'show only <name>', 'add all'. Examples: "
                    "'compare this to harbor seal' → SPECIES: add harbor seal, "
                    "'remove orca from the plot' → SPECIES: remove orca, "
                    "'show only gray whale' → SPECIES: show only gray whale, "
                    "'add all species' → SPECIES: add all, "
                    "'add bottlenose dolphin' → SPECIES: add bottlenose dolphin, "
                    "'add polar bear' → SPECIES: add polar bear, "
                    "'add hippo' → SPECIES: add hippopotamus. "
                    "Output 'SPECIES: ' followed by the action phrase, nothing else.\n"
                    "- CLUSTER_SEARCH: <protein or domain name> — use when the user asks WHICH "
                    "clusters contain a specific protein, domain, or function: 'which clusters have "
                    "myoglobin', 'find clusters with globin', 'show me clusters containing hemoglobin', "
                    "'where is rhodopsin distributed', 'in which clusters does MHC appear'. "
                    "Output just the search term after CLUSTER_SEARCH:, nothing else.\n"
                    "- HIGHLIGHT: <search term> — use ONLY when the user uses an explicit action "
                    "verb to locate something on the plot: 'highlight cluster 93', 'show me cluster 5 "
                    "on the plot', 'mark cluster 7', 'highlight protein A0A2U3', 'where is myoglobin "
                    "on the plot?', 'point to cluster 12'. Do NOT use HIGHLIGHT for questions like "
                    "'tell me more about cluster X', 'what is in cluster X', 'explain cluster X' — "
                    "those are QUERY or VISUAL. Only use HIGHLIGHT when the user's clear intent is to "
                    "move the visual focus, not to get information. Output just the cluster number, "
                    "Entry ID, or protein name after HIGHLIGHT:, nothing else.\n"
                    "- VISUAL: <the question> — use when the user wants you to DESCRIBE what's "
                    "on screen: 'what do I see?', 'explain this cluster', 'what do these proteins "
                    "have in common?', 'what's in the top-left?'. Pure observation questions.\n"
                    "- QUERY: <standalone question that includes relevant screen context> — use "
                    "when the user asks WHY something is the case, wants a deeper explanation, "
                    "or asks a biology/science question. This includes both general questions "
                    "('what is a proteome?') AND questions about on-screen data that need "
                    "knowledge to answer ('why does gray whale have no unique proteins?', "
                    "'why do these species overlap?', 'what does this domain do?'). "
                    "Include relevant context from the screen in your query so the knowledge "
                    "base can give a targeted answer.\n"
                    "- CHAT: <brief friendly reply in the voice of a warm, curious scientist "
                    "who is genuinely interested but not over the top — no 'Wow!', 'Oh awesome!', "
                    "or hollow exclamations. React naturally and move the conversation forward.> "
                    "— conversational (greeting, reaction, thanks)\n"
                    "Output only one line starting with CONTROL:, SPECIES:, CLUSTER_SEARCH:, HIGHLIGHT:, QUERY:, VISUAL:, or CHAT:"
                )},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            max_tokens=200,
        )
        result = resp.choices[0].message.content.strip()
        if result.startswith("CONTROL:"):
            return ("control", result[8:].strip())
        elif result.startswith("SPECIES:"):
            return ("species", result[8:].strip())
        elif result.startswith("CLUSTER_SEARCH:"):
            return ("cluster_search", result[15:].strip())
        elif result.startswith("HIGHLIGHT:"):
            return ("highlight", result[10:].strip())
        elif result.startswith("QUERY:"):
            return ("rag", result[6:].strip())
        elif result.startswith("VISUAL:"):
            return ("visual", result[7:].strip())
        elif result.startswith("CHAT:"):
            return ("chat", result[5:].strip())
        else:
            return ("rag", new_message)
    except Exception:
        return ("rag", new_message)


def answer_with_context(question, visual_context, chat_history):
    """Answer a question about what the user sees on screen using GPT directly."""
    messages = [
        {"role": "system", "content": (
            _APP_CONTEXT + "\n\n"
            "You are a nerdy scientist who is genuinely fascinated by proteins and what's on screen. "
            "You are the user's personal lab guide on this app. "
            "You find this stuff interesting and that comes through naturally — but you don't pepper "
            "responses with 'Wow!', 'Amazing!', or hollow enthusiasm. Let the science speak.\n\n"
            "CRITICAL — stay grounded in the data:\n"
            "- ONLY state things directly supported by the data provided to you.\n"
            "- Describe WHAT you see, don't fabricate WHY. If you don't know, say so honestly.\n\n"
            "Style: approachable language, 3-4 sentences max, **bold** key terms, suggest ONE next step."
        )}
    ]
    for msg in (chat_history or [])[-14:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"On my screen:\n{visual_context}\n\nQuestion: {question}"})
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def _build_sources_block(retrieved):
    """Build a deduplicated sources string from a list of Chroma Document objects."""
    if not retrieved:
        return ""
    _EMM_BOOK = "Encyclopedia of Marine Mammals (3rd Ed.)"
    doc_pages   = {}
    doc_ids     = {}
    doc_display = {}
    for d in retrieved:
        meta = d.metadata
        doc = meta.get("document_name", "").strip()
        if not doc:
            continue
        chunk_id = meta.get("id", "")
        page = str(meta.get("page_range") or meta.get("page_number") or "").strip()
        chapter = meta.get("chapter", "").strip()
        if chunk_id.startswith("html_"):
            key = f"html::{doc}"
            if key not in doc_display:
                chap_str = f"{chapter}: " if chapter and chapter not in ("Unknown Chapter", "") else ""
                doc_display[key] = f"*{_EMM_BOOK}* — {chap_str}{doc}"
        else:
            key = doc
            doc_display.setdefault(key, doc)
        doc_pages.setdefault(key, set())
        if page:
            doc_pages[key].add(page)
        doc_ids.setdefault(key, chunk_id)

    def _guess_doi(doc_name, chunk_id):
        m = re.search(r'elife[_\s](\d+)', chunk_id, re.IGNORECASE) or \
            re.search(r'elife[_\s](\d+)', doc_name, re.IGNORECASE)
        if m:
            return f"https://doi.org/10.7554/eLife.{m.group(1)}"
        m = re.search(r'srep(\d+)', chunk_id, re.IGNORECASE) or \
            re.search(r'srep(\d+)', doc_name, re.IGNORECASE)
        if m:
            return f"https://doi.org/10.1038/srep{m.group(1)}"
        return None

    source_lines = []
    for key, pages in doc_pages.items():
        page_str = ", ".join(
            sorted(pages, key=lambda p: int(p.split('-')[0]) if p.split('-')[0].isdigit() else 0)
        ) if pages else ""
        label = doc_display.get(key, key)
        doi = _guess_doi(key, doc_ids.get(key, ""))
        line = f"[{label}]({doi})" if doi else label
        if page_str:
            line += f" (p. {page_str})"
        source_lines.append(line)
    return "📚 " + " · ".join(source_lines) if source_lines else ""


def query_rag(query_text, visual_context=""):
    """Retrieve relevant chunks from Chroma and generate an answer via GPT-4o-mini."""
    try:
        vs = _load_vectorstore()
        retrieved = vs.similarity_search(query_text, k=RAG_TOP_K)

        if not retrieved:
            context = ""
        else:
            context = "\n\n---\n\n".join(d.page_content for d in retrieved)

        system_content = (
            "You are a marine biology knowledge assistant embedded inside the Marine Mammal Proteome Explorer, "
            "a tool for visualising and exploring the proteomes of seven marine and semi-aquatic mammals: "
            "gray whale (Eschrichtius robustus), killer whale (Orcinus orca), California sea lion (Zalophus californianus), "
            "harbor seal (Phoca vitulina), bottlenose dolphin (Tursiops truncatus), "
            "polar bear (Ursus maritimus), and hippopotamus (Hippopotamus amphibius).\n\n"
            "Tool methodology: Protein sequences were embedded with ProtT5-XL, a transformer-based protein language model "
            "pre-trained on UniRef50 that encodes structural, functional, and evolutionary properties into fixed-length vectors. "
            "A single shared UMAP was computed across all seven species (n_neighbors=50, min_dist=0.0) so positions are directly comparable. "
            "UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique: proteins that are "
            "functionally or evolutionarily similar end up close together in the 2D plot, while dissimilar proteins are far apart. "
            "Clustering was performed with K-means (K=370, chosen by silhouette score optimisation) on the 2D UMAP coordinates. "
            "There are 370 clusters in total, each named by the most frequent bi-gram in the protein names of that cluster "
            "(e.g. 'Olfactory Receptor', 'Large Ribosomal', 'Initiation Factor'). "
            "Species enrichment per cluster is assessed with a hypergeometric test on species membership with "
            "Benjamini-Hochberg FDR correction (padj < 0.05). A cluster is enriched for species X when X contributes "
            "significantly more proteins to that cluster than expected given its proteome size. "
            "20 out of 370 clusters show statistically significant enrichment: "
            "Polar Bear dominates 4 olfactory receptor clusters (massive OR gene family expansion, biologically meaningful); "
            "Sea Lion dominates 12 clusters (ribosomal, initiation factor, ATP synthase, NADH dehydrogenase — energy metabolism); "
            "Harbor Seal has 1 enriched cluster (Protocadherin Gamma); Orca has 2 (Low Quality artifacts from RefSeq annotation). "
            "Cetaceans (bottlenose dolphin, gray whale) show zero enrichment because their aquatic adaptation involved "
            "gene loss (olfactory receptor pseudogenization) rather than gene expansion — their signal is depletion, not enrichment. "
            "Most clusters (350/370) contain proteins from all 7 species, reflecting deep shared evolutionary heritage.\n\n"
            "Visualisation colours: each species has a fixed colour — "
            "California Sea Lion: red (#E6194B), Bottlenose Dolphin: blue (#4363D8), "
            "Gray Whale: green (#3CB44B), Orca: magenta (#F032E6), Harbor Seal: brown (#9A6324), "
            "Polar Bear: cyan (#46F0F0), Hippopotamus: orange (#F58231). "
            "Each dot in the UMAP represents one protein from one species. "
            "Cluster hulls (convex boundaries around clusters) are coloured by enriched species: "
            "an enriched cluster's hull uses that species' colour at higher opacity; "
            "non-enriched clusters have neutral gray hulls at low opacity. "
            "Centroid labels show the cluster name (not number) and are coloured by enriched species. "
            "Highlighted proteins are shown with gold rings.\n\n"
            "You have two sources of information:\n"
            "1. SCREEN DATA: What the user currently sees in the proteome visualizer "
            "(species displayed, UMAP clusters, selected proteins with IDs, domains, etc.). "
            "Use this to answer questions about specific clusters, proteins, or patterns "
            "visible in the current session.\n"
            "2. LITERATURE: A curated corpus of marine mammal research papers and species accounts "
            "(context chunks below). Each chunk starts with its source location, so cite naturally "
            "(e.g. 'According to Smith 2023, page 4...').\n"
            "Prioritise SCREEN DATA for questions about specific proteins, clusters, or "
            "what is currently displayed. Use LITERATURE for biology background and mechanisms. "
            "If neither source covers the question, answer from general marine biology knowledge "
            "but note: 'This is from general knowledge, not the specific literature.' "
            "Never fabricate citations or paper titles."
        )

        sources_block = _build_sources_block(retrieved)

        user_parts = []
        if visual_context:
            user_parts.append(f"Screen data (current visualization):\n{visual_context}")
        if context:
            user_parts.append(f"Literature context:\n{context}")
        user_parts.append(f"Question: {query_text}\n\nAnswer:")

        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ],
            temperature=0,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip(), sources_block
    except Exception as e:
        return f"Error: {str(e)}", ""


def rewrite_as_guide(raw_answer, original_question, chat_history, visual_context=""):
    """Rewrite a raw RAG answer in the enthusiastic scientist guide voice and suggest follow-up topics."""
    if raw_answer.startswith("Error:") or raw_answer.startswith("The knowledge base service"):
        return raw_answer
    try:
        history_ctx = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in (chat_history or [])[-14:]
        )
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    _APP_CONTEXT + "\n\n"
                    "You are a nerdy scientist — the user's personal lab guide on this app. "
                    "Rewrite the provided answer so it:\n"
                    "- Uses your warm, curious voice. Your interest is genuine, not performative — "
                    "avoid filler exclamations like 'Wow!', 'Amazing!', 'Oh cool!'. If something is "
                    "genuinely interesting, say *why* it's interesting instead of just reacting.\n"
                    "- Is in approachable, high-school-level language. Briefly explain any jargon.\n"
                    "- Stays concise: 3-4 sentences max for the explanation.\n"
                    "- If screen context is provided, connect the answer to what the user sees "
                    "(e.g. 'Looking at your plot, you can see...'). But only reference things "
                    "that are actually in the screen data.\n"
                    "- Uses markdown: **bold** for key terms, *italic* for emphasis, "
                    "emojis for personality (\U0001F9EC, \U0001F52C, \U0001F433, \U0001F3AF). "
                    "Use line breaks between thoughts.\n"
                    "- Ends with 1-2 follow-up topic suggestions phrased as friendly questions, "
                    "e.g. \"Want me to explain how protein folding works?\" or "
                    "\"Curious about how dolphins and whales compare at the protein level?\". "
                    "Pick topics that naturally flow from what was just discussed.\n"
                    "- Do NOT invent facts, evolutionary explanations, or causal reasoning "
                    "beyond what the source answer says. Only rephrase and simplify the source. "
                    "If the source doesn't explain why, don't make up a reason — just describe "
                    "what it says and suggest related topics to explore."
                )},
                {"role": "user", "content": (
                    f"Recent conversation:\n{history_ctx}\n\n"
                    + (f"What's on screen:\n{visual_context}\n\n" if visual_context else "")
                    + f"User asked: {original_question}\n\n"
                    f"Raw source answer:\n{raw_answer}\n\n"
                    "Rewrite this in your guide voice with follow-up topic suggestions."
                )}
            ],
            temperature=0.5,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return raw_answer


def chat_reply(message, chat_history, visual_context=""):
    """Handle conversational messages (greetings, thanks, follow-ups) with full history context."""
    messages = [
        {"role": "system", "content": (
            _APP_CONTEXT + "\n\n"
            "You are a nerdy scientist — the user's personal lab guide on this app. "
            "You're warm and curious but don't overreact — avoid 'Wow!', 'Amazing!', hollow exclamations. "
            "Express interest through what you say, not exclamation marks.\n"
            "Keep replies concise (2-3 sentences). "
            "Use **bold** key terms, *italics* for emphasis, occasional emojis (🧬 🔬 🐬 🎯). "
            "Only state things supported by the data on screen or conversation history. "
            "If you don't know why something is the way it is, say so and suggest how to investigate."
        )}
    ]
    for msg in (chat_history or [])[-14:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    user_content = message
    if visual_context:
        user_content = f"[What's on my screen right now:\n{visual_context}]\n\n{message}"
    messages.append({"role": "user", "content": user_content})
    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return message


# Ocean-inspired color palette
OCEAN_COLORS = {
    'deep_ocean': '#0a192f',
    'midnight': '#112240',
    'surface': '#1d3557',
    'wave': '#457b9d',
    'seafoam': '#a8dadc',
    'foam': '#f1faee',
    'coral': '#e63946',
    'kelp': '#2d6a4f',
    'bioluminescent': '#64ffda',
    'sunlight': '#ffd166',
}

# Species configuration - colors match animation for visual consistency
SPECIES_DATA = {
    'Sea Lion': {
        'csv': str(BASE_DIR / 'umap_output/California Sealion_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/sealion.png'),
        'color': '#E6194B',
        'glow': 'rgba(230, 25, 75, 0.4)',
        'display_name': 'California Sea Lion',
        'key': 'sealion'
    },
    'Bottlenose Dolphin': {
        'csv': str(BASE_DIR / 'umap_output/Bottlenose Dolphin_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/bottlenose.png'),
        'color': '#4363D8',
        'glow': 'rgba(67, 99, 216, 0.4)',
        'display_name': 'Bottlenose Dolphin',
        'key': 'bottlenose'
    },
    'Gray Whale': {
        'csv': str(BASE_DIR / 'umap_output/Graywhale_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/graywhale.png'),
        'color': '#3CB44B',
        'glow': 'rgba(60, 180, 75, 0.4)',
        'display_name': 'Gray Whale',
        'key': 'graywhale'
    },
    'Orca': {
        'csv': str(BASE_DIR / 'umap_output/Killer whale_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/orca.png'),
        'color': '#F032E6',
        'glow': 'rgba(240, 50, 230, 0.4)',
        'display_name': 'Orca (Killer Whale)',
        'key': 'orca'
    },
    'Harbor Seal': {
        'csv': str(BASE_DIR / 'umap_output/Harborseal_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/harbor_seal.png'),
        'color': '#9A6324',
        'glow': 'rgba(154, 99, 36, 0.4)',
        'display_name': 'Harbor Seal',
        'key': 'harborseal'
    },
    'Polar Bear': {
        'csv': str(BASE_DIR / 'umap_output/Polar Bear_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/polarbear.png'),
        'color': '#46F0F0',
        'glow': 'rgba(70, 240, 240, 0.4)',
        'display_name': 'Polar Bear',
        'key': 'polarbear'
    },
    'Hippopotamus': {
        'csv': str(BASE_DIR / 'umap_output/Hippopotamus_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/hippo.png'),
        'color': '#F58231',
        'glow': 'rgba(245, 130, 49, 0.4)',
        'display_name': 'Hippopotamus',
        'key': 'hippo'
    },
}

SAMPLE_SIZE = 3000
PRERENDERED_DIR = BASE_DIR / 'prerendered_animations'

# Per-cluster species enrichment (hypergeometric, BH-corrected).
# Cluster Label → {Dominant Species, Dominant Color, All Enriched, Cluster Name}
def _load_cluster_enrichment():
    path = BASE_DIR / 'umap_output' / 'cluster_enrichment.csv'
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return df.set_index('Cluster Label').to_dict('index')

_CLUSTER_ENRICHMENT: dict = _load_cluster_enrichment()

# Full-resolution search index: species_name -> list of {Entry, Protein names, Cluster Label}
# Populated when species are loaded; never sampled so protein searches are exhaustive.
_FULL_SEARCH_DATA: dict = {}

# Annotated CSV files per species (for protein classification/domain info)
ANNOTATION_FILES = {
    'bottlenose': [str(BASE_DIR / 'annotated/bottlenose_part1_annotated.csv'), str(BASE_DIR / 'annotated/bottlenose_part2_annotated.csv')],
    'graywhale': [str(BASE_DIR / 'annotated/graywhale_part1_annotated.csv'), str(BASE_DIR / 'annotated/graywhale_part2_annotated.csv')],
    'harborseal': [str(BASE_DIR / 'annotated/harborseal_part1_annotated.csv'), str(BASE_DIR / 'annotated/harborseal_part2_annotated.csv')],
    'orca': [str(BASE_DIR / 'annotated/orca_part1_annotated.csv'), str(BASE_DIR / 'annotated/orca_part2_annotated.csv')],
    'sealion': [str(BASE_DIR / 'annotated/sealion_part1_annotated.csv'), str(BASE_DIR / 'annotated/sealion_part2_annotated.csv')],
    'polarbear': [str(BASE_DIR / 'annotated/polarbear_part1_annotated.csv'), str(BASE_DIR / 'annotated/polarbear_part2_annotated.csv')],
    'hippo': [str(BASE_DIR / 'annotated/hippo_part1_annotated.csv'), str(BASE_DIR / 'annotated/hippo_part2_annotated.csv')],
}

def load_annotations():
    """Load annotation data from all annotated CSVs into a global lookup dict."""
    cols = ['ident', 'desc', 'sequence', 'clipDescription',
            'hmm_labels', 'hmm_pfam_ids', 'hmm_ranges', 'hmm_descriptions',
            'annotationConfidence', 'percentIdentity', 'queryCoverage',
            'order', 'family', 'genus']
    value_cols = [c for c in cols if c != 'ident']
    frames = []
    for file_list in ANNOTATION_FILES.values():
        for csv_path in file_list:
            p = Path(csv_path)
            if not p.exists():
                print(f"Warning: annotation file not found: {csv_path}")
                continue
            try:
                frames.append(pd.read_csv(p, usecols=cols))
            except Exception as e:
                print(f"Warning: failed to load {csv_path}: {e}")
    if not frames:
        print("Warning: no annotation files loaded")
        return {}
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=['ident'])
    df['ident'] = df['ident'].astype(str)
    df[value_cols] = df[value_cols].fillna('')
    for c in value_cols:
        df[c] = df[c].astype(str)
    df = df.set_index('ident')
    lookup = df[value_cols].to_dict('index')
    print(f"Loaded {len(lookup)} annotation entries")
    return lookup

# Load annotations at startup
ANNOTATIONS = load_annotations()

# ============= DATA LOADING =============

def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lower()
        if ext == '.gif':
            mime_type = 'image/gif'
        elif ext == '.png':
            mime_type = 'image/png'
        elif ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        return f"data:{mime_type};base64,{encoded}"
    except:
        return None

def get_animation_path(species_list):
    """Check if pre-rendered animation exists for species combination."""
    species_keys = sorted([SPECIES_DATA[species]['key'] for species in species_list])
    filename = '_'.join(species_keys) + '.gif'
    prerendered_path = PRERENDERED_DIR / filename
    if prerendered_path.exists():
        print(f"Using pre-rendered animation: {prerendered_path}")
        return str(prerendered_path), True
    print(f"Pre-rendered animation not found, will generate: {filename}")
    return str(prerendered_path), False

def generate_animation_on_demand(species_list):
    """Generate animation on-demand if not pre-rendered."""
    species_keys = [SPECIES_DATA[species]['key'] for species in species_list]
    animation_path, is_prerendered = get_animation_path(species_list)
    if is_prerendered:
        return animation_path
    PRERENDERED_DIR.mkdir(exist_ok=True, parents=True)
    cmd = [sys.executable, str(BASE_DIR / 'animal_morph_fixed.py')] + species_keys + ['--output', animation_path]
    print(f"Generating animation: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and Path(animation_path).exists():
            print(f"Successfully generated: {animation_path}")
            return animation_path
        else:
            print(f"Failed to generate animation: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("Animation generation timed out (>120s)")
        return None
    except Exception as e:
        print(f"Error generating animation: {str(e)}")
        return None

def process_image(fn, num_points):
    """Extract sampled coordinates and colors from animal silhouette."""
    image = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image file: {fn}")
    if image.ndim == 2:
        image_original = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_gray = image.copy()
        alpha = None
    elif image.shape[2] == 4:
        bgr = image[..., :3]
        alpha = image[..., 3]
        image_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        alpha = None
    if alpha is not None and np.std(alpha) > 1:
        mask = (alpha > 20).astype(np.uint8) * 255
    else:
        _, mask = cv2.threshold(image_gray, 40, 255, cv2.THRESH_BINARY)
        if np.mean(image_gray[mask == 255]) < np.mean(image_gray[mask == 0]):
            mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
    largest_contour = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, [largest_contour], -1, 255, cv2.FILLED)
    white_coords = np.column_stack(np.where(filled_mask == 255))
    if white_coords.shape[0] < num_points:
        raise ValueError(f"Not enough pixels ({white_coords.shape[0]} < {num_points})")
    np.random.seed(0)
    sampled_coords = white_coords[np.random.choice(len(white_coords), num_points, replace=False)]
    sampled_colors = image_original[sampled_coords[:, 0], sampled_coords[:, 1]]
    sampled_colors_normalized = sampled_colors / 255.0
    return sampled_coords, sampled_colors_normalized

def place_animals_smart(animals_data, plot_bounds):
    """Position animals in grid layout."""
    n_animals = len(animals_data)
    placed = []
    if n_animals <= 2:
        rows, cols = 1, 2
    elif n_animals <= 4:
        rows, cols = 2, 2
    else:
        rows = int(np.ceil(np.sqrt(n_animals)))
        cols = int(np.ceil(n_animals / rows))
    plot_x_min, plot_x_max, plot_y_min, plot_y_max = plot_bounds
    cell_width = (plot_x_max - plot_x_min) / cols
    cell_height = (plot_y_max - plot_y_min) / rows
    padding = 0.1
    usable_width = cell_width * (1 - 2 * padding)
    usable_height = cell_height * (1 - 2 * padding)
    for idx, animal in enumerate(animals_data):
        row = idx // cols
        col = idx % cols
        cell_center_x = plot_x_min + (col + 0.5) * cell_width
        cell_center_y = plot_y_max - (row + 0.5) * cell_height
        coords = animal["coords"]
        orig_width = coords[:, 1].max() - coords[:, 1].min()
        orig_height = coords[:, 0].max() - coords[:, 0].min()
        aspect_ratio = orig_width / orig_height
        if aspect_ratio > (usable_width / usable_height):
            scale = usable_width / orig_width
        else:
            scale = usable_height / orig_height
        scaled_coords_x = (coords[:, 1] - coords[:, 1].min()) * scale
        scaled_coords_y = (coords[:, 0] - coords[:, 0].min()) * scale
        final_x = scaled_coords_x - scaled_coords_x.mean() + cell_center_x
        final_y = -(scaled_coords_y - scaled_coords_y.mean()) + cell_center_y
        placed.append({
            "name": animal["name"],
            "x": final_x,
            "y": final_y,
            "colors": animal["colors"],
            "df": animal["df"],
            "color": animal["color"]
        })
    return placed

def load_data_for_species(species_list):
    """Load proteome and animal data for selected species."""
    PLOT_X_MIN, PLOT_X_MAX = -300, 300
    PLOT_Y_MIN, PLOT_Y_MAX = -300, 300
    animals_data = []
    for species in species_list:
        info = SPECIES_DATA[species]
        csv_path = Path(info['csv'])
        img_path = Path(info['image'])
        if not csv_path.exists():
            print(f"ERROR: CSV file not found for {species}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        essential_cols = ['Entry', 'Protein names', 'Gene Names', 'Organism',
                         'UMAP 1', 'UMAP 2', 'Cluster Label', 'Cluster Name', 'Length']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]
        # Store full (unsampled) data for exhaustive protein search before downsampling
        search_cols = [c for c in ['Entry', 'Protein names', 'Cluster Label', 'UMAP 1', 'UMAP 2'] if c in df.columns]
        _FULL_SEARCH_DATA[species] = df[search_cols].to_dict('records')
        if len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        if img_path.exists():
            try:
                coords, colors = process_image(str(img_path), len(df))
            except Exception as e:
                print(f"ERROR: Failed to process image for {species}: {e}")
                continue
        else:
            print(f"ERROR: Image file not found for {species}: {img_path}")
            continue
        animals_data.append({
            "name": species,
            "coords": coords,
            "colors": colors,
            "df": df,
            "color": info['color']
        })
    if not animals_data:
        raise ValueError("No valid animal data could be loaded.")
    plot_bounds = (PLOT_X_MIN, PLOT_X_MAX, PLOT_Y_MIN, PLOT_Y_MAX)
    placed_animals = place_animals_smart(animals_data, plot_bounds)
    all_dfs = [p["df"] for p in placed_animals]
    umap_x_min = min(df['UMAP 1'].min() for df in all_dfs)
    umap_x_max = max(df['UMAP 1'].max() for df in all_dfs)
    umap_y_min = min(df['UMAP 2'].min() for df in all_dfs)
    umap_y_max = max(df['UMAP 2'].max() for df in all_dfs)
    umap_center_x = (umap_x_min + umap_x_max) / 2
    umap_center_y = (umap_y_min + umap_y_max) / 2
    umap_max_range = max(umap_x_max - umap_x_min, umap_y_max - umap_y_min)
    scale_factor = (PLOT_X_MAX - PLOT_X_MIN) * 0.8 / umap_max_range
    for p in placed_animals:
        p["df"]['UMAP 1 Scaled'] = (p["df"]['UMAP 1'] - umap_center_x) * scale_factor
        p["df"]['UMAP 2 Scaled'] = (p["df"]['UMAP 2'] - umap_center_y) * scale_factor
    return placed_animals

# Load animal icons
print("Loading animal icons...")
animal_icons = {}
for species, info in SPECIES_DATA.items():
    img_path = Path(info['image'])
    if img_path.exists():
        animal_icons[species] = encode_image_to_base64(str(img_path))
print(f"Loaded {len(animal_icons)} animal icons")

# ============= DASH APP =============

app = dash.Dash(__name__)

# Serve prerendered animation files over HTTP (avoids 8MB+ base64 through WebSocket)
from flask import send_from_directory

@app.server.route('/animations/<path:filename>')
def serve_animation(filename):
    return send_from_directory(str(PRERENDERED_DIR), filename)

@app.server.route('/animal_pics/<path:filename>')
def serve_animal_pic(filename):
    return send_from_directory(str(BASE_DIR / 'animal_pics'), filename)



# Ocean theme CSS with waves, particles, and glassmorphism
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            * { box-sizing: border-box; }
            body {
                margin: 0;
                padding: 0;
                background: linear-gradient(180deg, #0a192f 0%, #112240 50%, #1d3557 100%);
                min-height: 100vh;
            }
            .ocean-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
                background: linear-gradient(180deg, #0a192f 0%, #112240 40%, #1d3557 100%);
            }
            .wave {
                position: absolute;
                bottom: 0;
                left: 0;
                width: 200%;
                height: 100px;
                background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'%3E%3Cpath fill='%23457b9d' fill-opacity='0.15' d='M0,192L48,197.3C96,203,192,213,288,229.3C384,245,480,267,576,250.7C672,235,768,181,864,181.3C960,181,1056,235,1152,234.7C1248,235,1344,181,1392,154.7L1440,128L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z'%3E%3C/path%3E%3C/svg%3E") repeat-x;
                animation: wave-animation 25s linear infinite;
            }
            .wave:nth-child(2) { bottom: 10px; opacity: 0.5; animation: wave-animation 20s linear reverse infinite; }
            .wave:nth-child(3) { bottom: 20px; opacity: 0.3; animation: wave-animation 30s linear infinite; }
            @keyframes wave-animation {
                0% { transform: translateX(0); }
                100% { transform: translateX(-50%); }
            }
            .particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                z-index: -1;
                pointer-events: none;
            }
            .particle {
                position: absolute;
                width: 6px;
                height: 6px;
                background: rgba(100, 255, 218, 0.3);
                border-radius: 50%;
                animation: float-up 15s infinite;
            }
            .particle:nth-child(1) { left: 10%; animation-delay: 0s; animation-duration: 20s; }
            .particle:nth-child(2) { left: 20%; animation-delay: 2s; animation-duration: 18s; }
            .particle:nth-child(3) { left: 30%; animation-delay: 4s; animation-duration: 22s; }
            .particle:nth-child(4) { left: 40%; animation-delay: 1s; animation-duration: 16s; }
            .particle:nth-child(5) { left: 50%; animation-delay: 3s; animation-duration: 24s; }
            .particle:nth-child(6) { left: 60%; animation-delay: 5s; animation-duration: 19s; }
            .particle:nth-child(7) { left: 70%; animation-delay: 2s; animation-duration: 21s; }
            .particle:nth-child(8) { left: 80%; animation-delay: 4s; animation-duration: 17s; }
            .particle:nth-child(9) { left: 90%; animation-delay: 1s; animation-duration: 23s; }
            @keyframes float-up {
                0% { transform: translateY(100vh) scale(0); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100px) scale(1); opacity: 0; }
            }
            .marine-loader {
                width: 80px;
                height: 80px;
                position: relative;
                margin: 0 auto 25px auto;
            }
            .marine-loader::before, .marine-loader::after {
                content: '';
                position: absolute;
                border-radius: 50%;
                animation: ripple 2s cubic-bezier(0, 0.2, 0.8, 1) infinite;
            }
            .marine-loader::before { width: 100%; height: 100%; border: 3px solid #64ffda; animation-delay: -0.5s; }
            .marine-loader::after { width: 100%; height: 100%; border: 3px solid #a8dadc; }
            @keyframes ripple {
                0% { transform: scale(0.1); opacity: 1; }
                100% { transform: scale(1); opacity: 0; }
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            }
            .glass-card:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(100, 255, 218, 0.3);
                box-shadow: 0 8px 32px rgba(100, 255, 218, 0.15);
                transform: translateY(-4px);
            }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; } }
            .fade-in { animation: fadeIn 0.8s ease-out; }
            .fade-out { animation: fadeOut 0.6s ease-in-out; }
            @keyframes plotEnter { from { opacity: 0; transform: scale(0.97); } to { opacity: 1; transform: scale(1); } }
            @keyframes hullPulse {
                0%   { opacity: 1; }
                50%  { opacity: 0.55; }
                100% { opacity: 1; }
            }
            #interactive-graph { animation: plotEnter 0.7s cubic-bezier(0.4,0,0.2,1); }
            #interactive-graph .js-plotly-plot .layer-above .scatterlayer .trace:first-child path {
                animation: hullPulse 4s ease-in-out infinite;
            }
            #animation-screen, #explorer-screen, #selection-screen { transition: opacity 0.8s ease-in-out; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #112240; }
            ::-webkit-scrollbar-thumb { background: #457b9d; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #64ffda; }
            .ocean-btn {
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            .ocean-btn::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                background: rgba(100, 255, 218, 0.2);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                transition: width 0.6s ease, height 0.6s ease;
            }
            .ocean-btn:hover::before { width: 300px; height: 300px; }
            .shimmer-text {
                background: linear-gradient(90deg, #a8dadc, #64ffda, #a8dadc);
                background-size: 200% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                animation: shimmer 3s linear infinite;
            }
            @keyframes shimmer {
                0% { background-position: 0% center; }
                100% { background-position: 200% center; }
            }
            .rc-slider-track { background: linear-gradient(90deg, #64ffda, #a8dadc) !important; }
            .rc-slider-handle { border-color: #64ffda !important; background-color: #64ffda !important; box-shadow: 0 0 10px rgba(100, 255, 218, 0.5) !important; }
            .rc-slider-rail { background-color: rgba(100, 255, 218, 0.2) !important; }
            .modebar, .js-plotly-plot .plotly .modebar {
                background: rgba(10, 25, 47, 0.85) !important;
                border-radius: 8px !important;
                padding: 4px 8px !important;
                left: 0 !important;
                right: auto !important;
            }
            .modebar-container {
                right: auto !important;
                left: 0 !important;
            }
            .modebar-btn { color: #a8dadc !important; }
            .modebar-btn:hover { color: #64ffda !important; }
            .modebar-btn.active { color: #64ffda !important; }
            /* Chat widget */
            .chat-toggle { position: fixed; bottom: 30px; right: 30px; width: 56px; height: 56px;
                border-radius: 50%; background: linear-gradient(135deg, #64ffda, #457b9d); border: none;
                cursor: pointer; box-shadow: 0 4px 20px rgba(100, 255, 218, 0.4); z-index: 1000;
                font-size: 14px; font-weight: 600; color: #0a192f; font-family: "Inter", sans-serif;
                transition: all 0.3s ease; letter-spacing: 0.5px; }
            .chat-toggle:hover { transform: scale(1.1); box-shadow: 0 6px 30px rgba(100, 255, 218, 0.6); }
            .chat-panel { position: fixed; bottom: 100px; right: 30px; width: 420px; height: 520px;
                background: rgba(10, 25, 47, 0.97); backdrop-filter: blur(20px);
                border: 1px solid rgba(100, 255, 218, 0.2); border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5); z-index: 999;
                display: flex; flex-direction: column; overflow: hidden; }
            .chat-header { padding: 16px 20px; background: rgba(100, 255, 218, 0.08);
                border-bottom: 1px solid rgba(100, 255, 218, 0.15); color: #64ffda;
                font-family: "JetBrains Mono", monospace; font-size: 12px;
                letter-spacing: 2px; text-transform: uppercase; font-weight: 500; }
            .chat-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex;
                flex-direction: column; gap: 12px; }
            .chat-msg { max-width: 85%; padding: 10px 14px; border-radius: 12px;
                font-family: "Inter", sans-serif; font-size: 13px; line-height: 1.5; word-wrap: break-word; }
            .chat-msg-user { align-self: flex-end; background: rgba(100, 255, 218, 0.15);
                color: #f1faee; border: 1px solid rgba(100, 255, 218, 0.2); border-bottom-right-radius: 4px; }
            .chat-msg-assistant { align-self: flex-start; background: rgba(255, 255, 255, 0.05);
                color: #a8dadc; border: 1px solid rgba(255, 255, 255, 0.1); border-bottom-left-radius: 4px; }
            .chat-md { margin: 0; }
            .chat-md p { margin: 0 0 8px 0; }
            .chat-md p:last-child { margin-bottom: 0; }
            .chat-md strong { color: #64ffda; font-weight: 600; }
            .chat-md em { color: #f1faee; font-style: italic; }
            .chat-md ul, .chat-md ol { margin: 4px 0; padding-left: 18px; }
            .chat-md li { margin-bottom: 2px; }
            .chat-md code { background: rgba(100, 255, 218, 0.1); padding: 1px 4px;
                border-radius: 3px; font-size: 12px; }
            .chat-input-area { padding: 12px 16px; border-top: 1px solid rgba(100, 255, 218, 0.15);
                display: flex; gap: 8px; align-items: center; }
            .chat-input-area input { flex: 1; padding: 10px 14px; background: #112240;
                border: 1px solid rgba(100, 255, 218, 0.2); border-radius: 10px; color: #f1faee;
                -webkit-text-fill-color: #f1faee; caret-color: #f1faee;
                font-family: "Inter", sans-serif; font-size: 13px; outline: none; }
            .chat-input-area input:focus { border-color: #64ffda; box-shadow: 0 0 10px rgba(100, 255, 218, 0.2); }
            .chat-input-area input::placeholder { color: #457b9d; }
            .chat-send-btn { padding: 10px 16px; background: rgba(100, 255, 218, 0.15); border: 1px solid #64ffda;
                border-radius: 10px; color: #64ffda; cursor: pointer; font-family: "JetBrains Mono", monospace;
                font-size: 12px; font-weight: 500; transition: all 0.2s ease; }
            .chat-send-btn:hover { background: rgba(100, 255, 218, 0.3); }
            /* Plot loading overlay */
            .plot-loading-overlay {
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: linear-gradient(180deg, #0a192f 0%, #112240 50%, #1d3557 100%);
                display: none; align-items: center; justify-content: center; flex-direction: column;
                z-index: 500; opacity: 0; transition: opacity 0.6s ease-out;
            }
            .plot-loading-overlay .loading-sub {
                color: #a8dadc; font-size: 13px; font-family: "JetBrains Mono", monospace;
                letter-spacing: 1px; margin-top: 10px;
            }
            /* AI hint toast */
            .ai-hint-toast {
                position: fixed; bottom: 96px; right: 30px;
                background: rgba(10, 25, 47, 0.95); backdrop-filter: blur(20px);
                border: 1px solid rgba(100, 255, 218, 0.3); border-radius: 12px;
                padding: 14px 20px; color: #a8dadc; font-family: "Inter", sans-serif; font-size: 13px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(100, 255, 218, 0.1);
                z-index: 998; display: none; opacity: 0; transform: translateY(10px);
                transition: opacity 0.5s ease, transform 0.5s ease;
                max-width: 280px; cursor: pointer; line-height: 1.5;
            }
            .ai-hint-toast:hover {
                border-color: #64ffda;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4), 0 0 30px rgba(100, 255, 218, 0.2);
            }
        </style>
    </head>
    <body>
        <div class="ocean-bg">
            <div class="wave"></div>
            <div class="wave"></div>
            <div class="wave"></div>
        </div>
        <div class="particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Store(id='loaded-data', data=None),
    dcc.Store(id='selected-species-store', data=[]),
    dcc.Store(id='active-species-store', data=[]),
    dcc.Store(id='chat-history', data=[]),
    dcc.Store(id='pending-query', data=None),

    # Screen 1: Species Selection
    html.Div([
        html.Div([
            html.Div([
                html.H1("Marine Mammal Proteome Explorer",
                       className='shimmer-text',
                       style={'textAlign': 'center', 'marginBottom': 8,
                             'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                             'fontSize': 48, 'letterSpacing': '2px'}),
                html.P("AI-Powered Proteomics Exploration",
                       style={'textAlign': 'center', 'color': '#64ffda', 'fontSize': 13,
                             'marginBottom': 8, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '3px', 'textTransform': 'uppercase'}),
                html.P("Select species to explore their proteomes, guided by an AI agent that explains what you see",
                       style={'textAlign': 'center', 'color': '#a8dadc', 'fontSize': 16,
                             'marginBottom': 60, 'fontFamily': '"Inter", sans-serif',
                             'fontWeight': '300'}),
            ], style={'marginBottom': 20}),

            # Hidden cards kept in DOM so per-species callbacks stay wired
            html.Div([
                html.Div([
                    html.Button('', id=f'btn-{species}', n_clicks=0,
                               style={'display': 'none'})
                ], id=f'card-{species}', style={'display': 'none'})
                for species, info in SPECIES_DATA.items()
            ], style={'display': 'none'}),

            # Phylogenetic tree + hover preview panel
            html.Div([
                # Tree (left, ~70%)
                html.Div([
                    html.P("Click a species to select · hover to preview · up to 3",
                           style={'textAlign': 'center', 'color': 'rgba(168,218,220,0.45)',
                                 'fontSize': 10, 'fontFamily': '"JetBrains Mono", monospace',
                                 'letterSpacing': '2px', 'marginBottom': 4,
                                 'textTransform': 'uppercase'}),
                    dcc.Graph(
                        id='selection-phylo-tree',
                        figure=go.Figure(),
                        config={'displayModeBar': False, 'scrollZoom': False},
                        style={'height': '580px'},
                    ),
                ], style={'flex': '1 1 65%', 'minWidth': 0}),

                # Hover preview panel (right, ~30%)
                html.Div([
                    html.Div(id='selection-hover-name',
                             children='Hover an animal',
                             style={'color': 'rgba(168,218,220,0.4)', 'fontSize': 11,
                                   'fontFamily': '"JetBrains Mono", monospace',
                                   'letterSpacing': '2px', 'textTransform': 'uppercase',
                                   'marginBottom': 16, 'textAlign': 'center'}),
                    html.Img(id='selection-hover-img', src='',
                             style={'maxWidth': '100%', 'maxHeight': '260px',
                                   'objectFit': 'contain', 'display': 'none',
                                   'borderRadius': 12,
                                   'filter': 'drop-shadow(0 0 20px rgba(100,255,218,0.3))'}),
                ], style={'flex': '0 0 28%', 'display': 'flex', 'flexDirection': 'column',
                         'alignItems': 'center', 'justifyContent': 'center',
                         'padding': '20px 10px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': 16,
                     'maxWidth': 1100, 'margin': '0 auto 40px auto'}),

            html.Div([
                html.Div([
                    html.H4("Selected Organisms",
                           style={'color': '#64ffda', 'marginBottom': 15,
                                 'fontSize': 12, 'fontFamily': '"JetBrains Mono", monospace',
                                 'letterSpacing': '2px', 'textTransform': 'uppercase'}),
                    html.Div(id='selected-species-display',
                            children="No organisms selected",
                            style={'color': '#a8dadc', 'fontSize': 15, 'marginBottom': 30,
                                  'fontFamily': '"Inter", sans-serif', 'minHeight': 30}),
                    html.Button('Analyze Proteomes', id='view-button', n_clicks=0,
                               className='ocean-btn',
                               style={'padding': '16px 48px', 'fontSize': 13,
                                     'backgroundColor': 'rgba(100, 255, 218, 0.1)',
                                     'color': '#64ffda',
                                     'border': '1px solid #64ffda',
                                     'borderRadius': 30,
                                     'cursor': 'pointer', 'fontWeight': '500',
                                     'transition': 'all 0.3s ease',
                                     'fontFamily': '"JetBrains Mono", monospace',
                                     'letterSpacing': '2px',
                                     'textTransform': 'uppercase'},
                               disabled=True)
                ], className='glass-card',
                   style={'padding': 40, 'textAlign': 'center',
                         'background': 'rgba(255, 255, 255, 0.02)',
                         'backdropFilter': 'blur(10px)',
                         'border': '1px solid rgba(255, 255, 255, 0.08)',
                         'borderRadius': 20,
                         'maxWidth': 600, 'margin': '0 auto'})
            ])
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50})
    ], id='selection-screen', style={'display': 'block', 'backgroundColor': 'transparent',
                                      'minHeight': '100vh', 'paddingTop': 60}),

    # Screen 2: Loading + Animation
    html.Div([
        html.Div([
            html.Div([
                html.Div(className='marine-loader'),
                html.H3(id='loading-message',
                       children="Preparing Proteome Visualization",
                       className='shimmer-text',
                       style={'marginBottom': 12,
                             'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                             'fontSize': 28}),
                html.P("Initializing multi-dimensional protein space transformation...",
                      style={'color': '#a8dadc', 'fontSize': 13,
                            'fontFamily': '"JetBrains Mono", monospace',
                            'letterSpacing': '1px'})
            ], id='loading-indicator', style={'textAlign': 'center', 'padding': '120px 20px'}),

            html.Div([
                html.H2("Proteome Space Transformation",
                       className='shimmer-text',
                       style={'textAlign': 'center', 'marginBottom': 10,
                             'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                             'fontSize': 36, 'letterSpacing': '1px'}),
                html.P("Watch as species silhouettes morph into their UMAP embeddings",
                      style={'textAlign': 'center', 'color': '#a8dadc', 'fontSize': 14,
                            'marginBottom': 40, 'fontFamily': '"JetBrains Mono", monospace',
                            'letterSpacing': '1px'}),
                html.Div([
                    html.Img(id='animation-gif', src='', style={
                        'maxWidth': '100%',
                        'maxHeight': '70vh',
                        'display': 'block',
                        'margin': '0 auto',
                        'borderRadius': '16px',
                        'boxShadow': '0 20px 60px rgba(0, 0, 0, 0.4), 0 0 40px rgba(100, 255, 218, 0.1)',
                        'border': '1px solid rgba(100, 255, 218, 0.2)',
                    })
                ], style={'textAlign': 'center'})
            ], id='animation-container', style={'display': 'none'})
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': 50})
    ], id='animation-screen', style={'display': 'none', 'backgroundColor': 'transparent',
                                     'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1}),

    # Screen 3: Interactive Explorer
    html.Div([
        # Slim horizontal toolbar
        html.Div([
            # Back button
            html.Button('←', id='back-button', n_clicks=0,
                       className='ocean-btn',
                       style={'padding': '6px 12px',
                             'backgroundColor': 'transparent', 'color': '#64ffda',
                             'border': '1px solid rgba(100, 255, 218, 0.3)',
                             'borderRadius': 8,
                             'cursor': 'pointer', 'fontFamily': '"JetBrains Mono", monospace',
                             'fontSize': 14, 'fontWeight': '500',
                             'lineHeight': '1', 'flexShrink': '0'}),

            # Separator
            html.Div(style={'width': 1, 'height': 28, 'background': 'rgba(100, 255, 218, 0.2)',
                           'flexShrink': '0'}),

            # Species toggle buttons
            html.Div([
                html.Button([
                    html.Img(src=animal_icons.get(species, ''),
                             style={'width': '22px', 'height': '22px', 'objectFit': 'contain',
                                    'filter': 'brightness(0) invert(1) opacity(0.9)',
                                    'flexShrink': '0'}),
                    html.Span(SPECIES_DATA[species]['display_name'],
                              style={'fontSize': 10, 'fontWeight': '500',
                                     'margin': 0, 'whiteSpace': 'nowrap',
                                     'fontFamily': '"Inter", sans-serif'})
                ], id=f'toggle-{species}', n_clicks=0,
                   style={'display': 'inline-flex', 'alignItems': 'center', 'gap': '4px',
                          'padding': '3px 8px', 'background': 'rgba(255, 255, 255, 0.05)',
                          'borderRadius': 8, 'border': '1px solid rgba(255,255,255,0.15)',
                          'cursor': 'pointer', 'opacity': 0.5,
                          'color': '#a8dadc', 'transition': 'all 0.2s ease'})
                for species in SPECIES_DATA
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '6px',
                      'flexWrap': 'wrap', 'flexShrink': '1', 'minWidth': '0'}),

            # Separator
            html.Div(style={'width': 1, 'height': 28, 'background': 'rgba(100, 255, 218, 0.2)',
                           'flexShrink': '0'}),

            # Phylogeny toggle
            html.Button('🌿 Phylogeny', id='phylo-toggle-btn', n_clicks=0,
                       style={'padding': '4px 10px', 'backgroundColor': 'transparent',
                              'color': '#64ffda', 'border': '1px solid rgba(100,255,218,0.3)',
                              'borderRadius': 8, 'cursor': 'pointer',
                              'fontFamily': '"JetBrains Mono", monospace',
                              'fontSize': 11, 'fontWeight': '500', 'flexShrink': '0'}),

            # Separator
            html.Div(style={'width': 1, 'height': 28, 'background': 'rgba(100, 255, 218, 0.2)',
                           'flexShrink': '0'}),

            # Point Size slider
            html.Div([
                html.Label("Size",
                          style={'fontWeight': '500', 'fontSize': 10,
                                'color': '#a8dadc', 'fontFamily': '"JetBrains Mono", monospace',
                                'marginRight': 4, 'whiteSpace': 'nowrap', 'letterSpacing': '0.5px'}),
                html.Div([
                    dcc.Slider(id='point-size', min=1, max=10, value=3, step=1,
                              marks={1: {'label': '1', 'style': {'color': '#a8dadc', 'fontSize': '9px'}},
                                    5: {'label': '5', 'style': {'color': '#a8dadc', 'fontSize': '9px'}},
                                    10: {'label': '10', 'style': {'color': '#a8dadc', 'fontSize': '9px'}}})
                ], style={'width': 120})
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'flexShrink': '0'}),

            # Opacity slider
            html.Div([
                html.Label("Opacity",
                          style={'fontWeight': '500', 'fontSize': 10,
                                'color': '#a8dadc', 'fontFamily': '"JetBrains Mono", monospace',
                                'marginRight': 4, 'whiteSpace': 'nowrap', 'letterSpacing': '0.5px'}),
                html.Div([
                    dcc.Slider(id='opacity', min=0.1, max=1.0, value=1.0, step=0.1,
                              marks={0.1: {'label': '.1', 'style': {'color': '#a8dadc', 'fontSize': '9px'}},
                                    0.5: {'label': '.5', 'style': {'color': '#a8dadc', 'fontSize': '9px'}},
                                    1.0: {'label': '1', 'style': {'color': '#a8dadc', 'fontSize': '9px'}}})
                ], style={'width': 120})
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px', 'flexShrink': '0'}),

            # Separator
            html.Div(style={'width': 1, 'height': 28, 'background': 'rgba(100, 255, 218, 0.2)',
                           'flexShrink': '0'}),

            # Protein filter radio (inline)
            html.Div([
                dcc.RadioItems(
                    id='protein-filter',
                    options=[
                        {'label': 'ALL', 'value': 'all'},
                        {'label': 'SPECIES-SPECIFIC', 'value': 'unique'},
                        {'label': 'SHARED', 'value': 'shared'}
                    ],
                    value='all',
                    style={'fontFamily': '"JetBrains Mono", monospace', 'fontSize': 10,
                           'letterSpacing': '0.5px', 'display': 'flex', 'gap': '2px'},
                    labelStyle={'display': 'inline-flex', 'alignItems': 'center',
                                'color': '#a8dadc', 'cursor': 'pointer',
                                'whiteSpace': 'nowrap'},
                    inputStyle={'marginRight': 4}
                ),
            ], id='protein-filter-container',
               style={'display': 'flex', 'alignItems': 'center', 'gap': '6px', 'flexShrink': '0'}),

            # Filter count
            html.Div(id='filter-count-display',
                     style={'fontSize': 10, 'color': '#457b9d',
                            'fontFamily': '"JetBrains Mono", monospace',
                            'letterSpacing': '0.5px', 'whiteSpace': 'nowrap', 'flexShrink': '0'})

        ], className='glass-card',
           style={'display': 'flex', 'alignItems': 'center', 'gap': '12px',
                 'padding': '8px 16px',
                 'background': 'rgba(10, 25, 47, 0.9)',
                 'backdropFilter': 'blur(15px)',
                 'borderRadius': 12,
                 'border': '1px solid rgba(100, 255, 218, 0.15)',
                 'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)',
                 'marginBottom': 10, 'flexWrap': 'wrap'}),

        # Phylogenetic tree panel (collapsible)
        html.Div(id='phylo-panel', children=[
            dcc.Graph(
                id='phylo-tree-graph',
                config={'displayModeBar': False, 'responsive': True},
                style={'height': '220px', 'width': '100%'},
            )
        ], style={
            'background': 'rgba(10, 25, 47, 0.85)',
            'borderRadius': 12,
            'border': '1px solid rgba(100, 255, 218, 0.12)',
            'padding': '4px 8px',
            'marginBottom': 10,
            'display': 'none',
        }),

        # Full-width Plot
        html.Div([
            dcc.Loading(
                dcc.Graph(id='interactive-graph',
                         style={'height': 'calc(100vh - 200px)', 'width': '100%'},
                         config={'displayModeBar': True, 'displaylogo': False,
                                 'responsive': True}),
                type='circle',
                color='#64ffda',
                style={'height': 'calc(100vh - 200px)'},
            )
        ], style={'background': '#ffffff',
                 'borderRadius': 16, 'padding': 10,
                 'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.2)',
                 'border': '1px solid rgba(100, 255, 218, 0.2)'}),

        # Protein Selection Table - Below the plot (full width)
        html.Div([
            html.Div([
                html.H3("Protein Table",
                       style={'color': '#64ffda', 'marginBottom': 0,
                             'fontSize': 12, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '2px', 'textTransform': 'uppercase',
                             'fontWeight': '500'}),
                html.Span("Lasso proteins on the plot, or search below to find and highlight them",
                         style={'fontSize': 12, 'color': '#a8dadc',
                               'fontFamily': '"JetBrains Mono", monospace',
                               'letterSpacing': '0.5px'}),
            ], style={'marginBottom': 14}),

            # Search bar — now lives here, clearly associated with the table
            html.Div([
                dcc.Input(
                    id='highlight-search-input',
                    type='text',
                    placeholder='Search by cluster # or protein name (e.g. "myoglobin", "93")...',
                    debounce=False,
                    n_submit=0,
                    style={
                        'flex': '1', 'background': 'rgba(255,255,255,0.06)',
                        'border': '1px solid rgba(100,255,218,0.2)', 'borderRadius': 8,
                        'color': '#f1faee', 'padding': '7px 12px', 'fontSize': 12,
                        'fontFamily': '"JetBrains Mono", monospace', 'outline': 'none',
                    }
                ),
                html.Button('🔍 Find & Highlight', id='highlight-search-btn', n_clicks=0, style={
                    'background': 'linear-gradient(135deg, #64ffda, #457b9d)',
                    'color': '#0a192f', 'border': 'none', 'borderRadius': 8,
                    'padding': '7px 14px', 'fontSize': 12, 'fontWeight': '700',
                    'cursor': 'pointer', 'fontFamily': '"Inter", sans-serif', 'whiteSpace': 'nowrap',
                }),
                html.Button('✕ Clear', id='highlight-clear-btn', n_clicks=0, style={
                    'background': 'rgba(255,255,255,0.06)',
                    'color': '#a8dadc', 'border': '1px solid rgba(100,255,218,0.15)',
                    'borderRadius': 8, 'padding': '7px 12px', 'fontSize': 12,
                    'cursor': 'pointer', 'fontFamily': '"Inter", sans-serif',
                }),
            ], style={
                'display': 'flex', 'alignItems': 'center', 'gap': '8px',
                'marginBottom': 16,
            }),

            html.Div(id='selection-table-container',
                    children="No proteins selected. Use the lasso tool on the plot, or search above.",
                    style={'fontSize': 13, 'fontFamily': '"Inter", sans-serif', 'color': '#a8dadc'})
        ], className='glass-card',
           style={'marginTop': 20, 'padding': 25,
                 'background': 'rgba(10, 25, 47, 0.7)',
                 'backdropFilter': 'blur(10px)',
                 'borderRadius': 16,
                 'border': '1px solid rgba(100, 255, 218, 0.1)',
                 'boxShadow': '0 15px 40px rgba(0, 0, 0, 0.2)'})

    ], id='explorer-screen', style={'display': 'none', 'fontFamily': '"Inter", sans-serif',
                                    'padding': '15px 20px', 'backgroundColor': 'transparent', 'opacity': 0}),

    dcc.Interval(id='gif-load-poller', interval=200, disabled=True, n_intervals=0),
    dcc.Interval(id='animation-timer', interval=200, disabled=True, n_intervals=0),
    dcc.Interval(id='fade-complete-timer', interval=800, disabled=True, n_intervals=0, max_intervals=1),
    dcc.Store(id='animation-path-store', data=None),
    dcc.Store(id='highlight-store', data=None),
    dcc.Store(id='last-click-store', data=None),
    dcc.Store(id='selected-cluster-store', data=None),

    # Chat widget - floating toggle + panel
    html.Button('Ask AI', id='chat-toggle-btn', n_clicks=0, className='chat-toggle'),
    html.Div([
        html.Div([
            html.Span("AI Agent", style={'flex': '1'}),
            html.Div(id='selected-cluster-badge', style={
                'fontSize': 11, 'color': '#64ffda', 'fontFamily': '"Inter", sans-serif',
                'opacity': 0.8,
            }),
        ], className='chat-header', style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
        html.Div(
            id='chat-messages-display',
            children=[
                html.Div(dcc.Markdown(
                    "Hey there! \U0001F30A Welcome to the **Proteome Explorer**! "
                    "Each dot on this plot is a *protein* — and when dots **cluster together**, "
                    "it means those proteins share similar structure or function.",
                    className='chat-md'),
                    className='chat-msg chat-msg-assistant'),
                html.Div(dcc.Markdown(
                    "Here's what you can do:\n\n"
                    "- \U0001F3AF **Lasso** proteins on the plot to see their details in the table below\n"
                    "- \U0001F50D **Search** for a cluster number or protein name using the search bar in the Protein Table\n"
                    "- \U0001F4AC **Ask me** anything — *\"what is myoglobin?\"*, *\"highlight cluster 14\"*, *\"compare to harbor seal\"*\n"
                    "- \U0001F40B **Add or remove species** by asking — *\"add orca\"*, *\"remove sea lion\"*, *\"show only gray whale\"*\n"
                    "- \U0001F3A8 **Control the plot** by asking — *\"show unique proteins\"*, *\"make the dots bigger\"*, *\"set opacity to 0.5\"*\n"
                    "- \U0001F4CC **Click a coloured hull** on the UMAP to select a cluster — I'll load its enrichment context and be ready to answer your questions about it!",
                    className='chat-md'),
                    className='chat-msg chat-msg-assistant'),
            ],
            className='chat-messages',
        ),
        html.Div([
            dcc.Input(id='chat-input', type='text',
                      placeholder='Ask about marine mammals...',
                      debounce=False, n_submit=0),
            html.Button('Send', id='chat-send-btn', n_clicks=0, className='chat-send-btn'),
        ], className='chat-input-area'),
    ], id='chat-panel', className='chat-panel', style={'display': 'none'}),

    # Plot loading overlay (between animation and interactive explorer)
    html.Div([
        html.Div(className='marine-loader'),
        html.H3("Rendering Proteome Map",
                 className='shimmer-text',
                 style={'fontFamily': '"Inter", sans-serif', 'fontWeight': '300',
                        'fontSize': 24, 'marginBottom': 0}),
        html.P("Almost there...", className='loading-sub'),
    ], id='plot-loading-overlay', className='plot-loading-overlay'),

    # AI hint toast (appears once plot is ready)
    html.Div([
        html.Span("AI",
                   style={'background': 'linear-gradient(135deg, #64ffda, #457b9d)',
                          'color': '#0a192f', 'borderRadius': '50%', 'width': '24px',
                          'height': '24px', 'display': 'inline-flex', 'alignItems': 'center',
                          'justifyContent': 'center', 'fontSize': '10px', 'fontWeight': '700',
                          'marginRight': '10px', 'flexShrink': '0',
                          'fontFamily': '"Inter", sans-serif'}),
        html.Span("Hey! I can walk you through what's on screen — click to chat!"),
    ], id='ai-hint-toast', className='ai-hint-toast'),

], style={'fontFamily': 'Arial, sans-serif'})

def _compute_filter_tags(species_data_list):
    """Tag proteins as unique (unclustered, Cluster Label == -1) or shared (in a cluster).

    With highly conserved mammalian proteomes, every HDBSCAN cluster contains proteins
    from all species. 'Unique' therefore means proteins that did NOT cluster with any
    other proteins — HDBSCAN noise points (label == -1). These represent functionally
    orphan proteins with no close analogs in the other species' proteomes.
    """
    for sd in species_data_list:
        for rec in sd['df']:
            cl = rec.get('Cluster Label')
            try:
                is_unclustered = cl is None or int(float(cl)) < 0
            except (TypeError, ValueError):
                is_unclustered = True
            rec['filter_tag'] = 'unique' if is_unclustered else 'shared'
    return species_data_list


# ============= CALLBACKS =============

for species in SPECIES_DATA.keys():
    @app.callback(
        [Output(f'card-{species}', 'style'),
         Output(f'btn-{species}', 'children'),
         Output(f'btn-{species}', 'style')],
        [Input(f'btn-{species}', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_species(n_clicks, species=species):
        selected = n_clicks % 2 == 1
        species_color = SPECIES_DATA[species]["color"]
        species_glow = SPECIES_DATA[species].get("glow", "rgba(100, 255, 218, 0.3)")

        card_style = {
            'padding': 25, 'textAlign': 'center',
            'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
            'background': 'rgba(255, 255, 255, 0.08)' if selected else 'rgba(255, 255, 255, 0.03)',
            'backdropFilter': 'blur(10px)',
            'border': f'1px solid {species_color}' if selected else '1px solid rgba(255, 255, 255, 0.1)',
            'borderRadius': 16,
            'boxShadow': f'0 8px 32px {species_glow}, 0 0 20px {species_glow}' if selected else '0 8px 32px rgba(0, 0, 0, 0.2)',
            'transform': 'translateY(-6px) scale(1.02)' if selected else 'translateY(0) scale(1)'
        }

        btn_text = 'Selected' if selected else 'Select'
        btn_style = {
            'padding': '10px 28px', 'fontSize': 11,
            'backgroundColor': species_color if selected else 'transparent',
            'color': '#0a192f' if selected else '#a8dadc',
            'border': f'1px solid {species_color}',
            'borderRadius': 25, 'cursor': 'pointer',
            'transition': 'all 0.3s ease', 'fontWeight': '500',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px', 'textTransform': 'uppercase'
        }

        return card_style, btn_text, btn_style

@app.callback(
    [Output('selected-species-display', 'children'),
     Output('selected-species-store', 'data'),
     Output('view-button', 'disabled'),
     Output('view-button', 'style')],
    [Input(f'btn-{species}', 'n_clicks') for species in SPECIES_DATA.keys()]
)
def update_selected_display(*n_clicks_list):
    selected = [species for species, clicks in zip(SPECIES_DATA.keys(), n_clicks_list)
                if clicks and clicks % 2 == 1]

    MAX_SPECIES = 3

    if not selected:
        display = "No organisms selected"
        disabled = True
        btn_style = {
            'padding': '16px 48px', 'fontSize': 13,
            'backgroundColor': 'rgba(100, 100, 100, 0.1)',
            'color': '#457b9d',
            'border': '1px solid rgba(100, 100, 100, 0.2)',
            'borderRadius': 30,
            'cursor': 'not-allowed', 'fontWeight': '500',
            'transition': 'all 0.3s ease',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px',
            'textTransform': 'uppercase'
        }
    elif len(selected) > MAX_SPECIES:
        display = html.Div([
            html.Div([
                html.Span([
                    html.Span("●", style={'marginRight': 8, 'fontSize': 10}),
                    f"{SPECIES_DATA[species]['display_name']}"
                ], style={'color': SPECIES_DATA[species]['color'],
                         'fontWeight': '500', 'marginRight': 20, 'fontSize': 15,
                         'fontFamily': '"Inter", sans-serif'})
                for species in selected
            ]),
            html.Div(f"Maximum {MAX_SPECIES} species — please deselect {len(selected) - MAX_SPECIES}",
                     style={'color': '#ff6b6b', 'fontSize': 12, 'marginTop': 8,
                            'fontFamily': '"JetBrains Mono", monospace'})
        ])
        disabled = True
        btn_style = {
            'padding': '16px 48px', 'fontSize': 13,
            'backgroundColor': 'rgba(100, 100, 100, 0.1)',
            'color': '#457b9d',
            'border': '1px solid rgba(255, 107, 107, 0.4)',
            'borderRadius': 30,
            'cursor': 'not-allowed', 'fontWeight': '500',
            'transition': 'all 0.3s ease',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px',
            'textTransform': 'uppercase'
        }
    else:
        display = html.Div([
            html.Span([
                html.Span("●", style={'marginRight': 8, 'fontSize': 10}),
                f"{SPECIES_DATA[species]['display_name']}"
            ], style={'color': SPECIES_DATA[species]['color'],
                     'fontWeight': '500', 'marginRight': 20, 'fontSize': 15,
                     'fontFamily': '"Inter", sans-serif',
                     'textShadow': f'0 0 10px {SPECIES_DATA[species].get("glow", "rgba(100, 255, 218, 0.3)")}'})
            for species in selected
        ])
        disabled = False
        btn_style = {
            'padding': '16px 48px', 'fontSize': 13,
            'backgroundColor': '#64ffda',
            'color': '#0a192f',
            'border': 'none',
            'borderRadius': 30,
            'cursor': 'pointer', 'fontWeight': '600',
            'boxShadow': '0 0 30px rgba(100, 255, 218, 0.4), 0 10px 40px rgba(100, 255, 218, 0.2)',
            'transition': 'all 0.3s ease',
            'fontFamily': '"JetBrains Mono", monospace',
            'letterSpacing': '2px',
            'textTransform': 'uppercase'
        }

    return display, selected, disabled, btn_style

@app.callback(
    [Output('selection-screen', 'style'),
     Output('animation-screen', 'style')],
    [Input('view-button', 'n_clicks')],
    [State('selected-species-store', 'data')],
    prevent_initial_call=True
)
def show_animation_screen(n_clicks, selected_species):
    if not selected_species:
        raise dash.exceptions.PreventUpdate
    selection_style = {'display': 'none'}
    animation_style = {'display': 'block', 'backgroundColor': 'transparent',
                      'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1}
    return selection_style, animation_style

@app.callback(
    [Output('loaded-data', 'data'),
     Output('active-species-store', 'data'),
     Output('animation-path-store', 'data'),
     Output('loading-message', 'children')],
    [Input('view-button', 'n_clicks')],
    [State('selected-species-store', 'data')],
    prevent_initial_call=True
)
def load_proteome_data(n_clicks, selected_species):
    if not selected_species:
        raise dash.exceptions.PreventUpdate
    # Animation uses only selected species
    animation_path, is_prerendered = get_animation_path(selected_species)
    if not is_prerendered:
        loading_msg = "Rendering Custom Proteome Visualization"
    else:
        loading_msg = "Preparing Proteome Visualization"
    if not is_prerendered:
        print(f"Generating animation on-demand for: {selected_species}")
        animation_path = generate_animation_on_demand(selected_species)
        if not animation_path:
            print("Failed to generate animation, using placeholder")
            animation_path = None
    # Load ALL species data so toggling is instant
    all_species = list(SPECIES_DATA.keys())
    print(f"Loading data for ALL species: {all_species}")
    placed_animals = load_data_for_species(all_species)

    data_json = []
    for p in placed_animals:
        data_json.append({
            'name': p['name'],
            'x': p['x'].tolist(),
            'y': p['y'].tolist(),
            'colors': p['colors'].tolist(),
            'color': p['color'],
            'df': p['df'].to_dict('records'),
            'umap_1_scaled': p['df']['UMAP 1 Scaled'].tolist(),
            'umap_2_scaled': p['df']['UMAP 2 Scaled'].tolist()
        })
    # Initialize active species to the user's original selection
    return data_json, selected_species, animation_path, loading_msg

@app.callback(
    [Output('loading-indicator', 'style'),
     Output('animation-container', 'style'),
     Output('animation-gif', 'src'),
     Output('gif-load-poller', 'disabled'),
     Output('explorer-screen', 'style', allow_duplicate=True)],
    [Input('loaded-data', 'data'),
     Input('animation-path-store', 'data')],
    prevent_initial_call=True
)
def display_animation(loaded_data, animation_path):
    if not loaded_data:
        raise dash.exceptions.PreventUpdate
    print(f"Data loaded! Displaying animation from: {animation_path}")
    if animation_path and Path(animation_path).exists():
        gif_url = f'/animations/{Path(animation_path).name}'
    else:
        gif_url = ''
        print("Warning: No animation available")
    loading_style = {'display': 'none'}
    animation_container_style = {'display': 'block'}
    # Render explorer invisibly in background so the plot builds while animation plays
    explorer_bg_style = {'display': 'block', 'fontFamily': '"Inter", sans-serif',
                         'padding': '15px 20px', 'backgroundColor': 'transparent', 'opacity': 0}
    # Start polling for GIF load; animation-timer stays disabled until GIF is ready
    return loading_style, animation_container_style, gif_url, False, explorer_bg_style

# Poll every 200ms until the GIF is loaded in the browser, then restart it
# from frame 1 and enable the 5s animation timer.
app.clientside_callback(
    """
    function(n) {
        var img = document.getElementById('animation-gif');
        if (!img || !img.src || !img.complete || img.naturalWidth === 0) {
            /* Not loaded yet — keep polling, keep animation-timer disabled */
            return [false, true];
        }
        /* GIF is loaded and playing — stop poller, start the 5s timer */
        return [true, false];
    }
    """,
    [Output('gif-load-poller', 'disabled', allow_duplicate=True),
     Output('animation-timer', 'disabled', allow_duplicate=True)],
    Input('gif-load-poller', 'n_intervals'),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(n_intervals) {
        if (n_intervals < 1) {
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }

        /* Check if Plotly graph has real data rendered */
        var graphEl = document.getElementById('interactive-graph');
        var plotDiv = graphEl ? graphEl.querySelector('.js-plotly-plot') : null;
        var ready = false;
        if (plotDiv && plotDiv.data && plotDiv.data.length > 0) {
            for (var i = 0; i < plotDiv.data.length; i++) {
                if (plotDiv.data[i].x && plotDiv.data[i].x.length > 0) {
                    ready = true;
                    break;
                }
            }
        }

        if (!ready) {
            /* Plot not ready — keep animation playing, keep polling */
            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }

        /* Plot is ready — stop polling and transition */
        var animScreen = document.getElementById('animation-screen');
        var explorerScreen = document.getElementById('explorer-screen');

        if (animScreen && explorerScreen) {
            /* Fade out animation, fade in explorer simultaneously */
            animScreen.style.transition = 'opacity 0.8s ease-out';
            animScreen.style.opacity = '0';
            explorerScreen.style.transition = 'opacity 0.8s ease-in';
            explorerScreen.style.opacity = '1';

            setTimeout(function() {
                animScreen.style.display = 'none';

                /* Move modebar to top-left and keep it there on re-renders */
                function fixModebar() {
                    document.querySelectorAll('.modebar').forEach(function(el) {
                        el.style.setProperty('right', 'auto', 'important');
                        el.style.setProperty('left', '2px', 'important');
                    });
                    document.querySelectorAll('.modebar-container').forEach(function(el) {
                        el.style.setProperty('right', 'auto', 'important');
                        el.style.setProperty('left', '0', 'important');
                    });
                }
                fixModebar();
                if (graphEl && !graphEl._modebarObserver) {
                    graphEl._modebarObserver = new MutationObserver(fixModebar);
                    graphEl._modebarObserver.observe(graphEl, {childList: true, subtree: true});
                }

                /* Show AI hint toast */
                setTimeout(function() {
                    var toast = document.getElementById('ai-hint-toast');
                    if (!toast) return;
                    toast.style.display = 'flex';
                    toast.style.alignItems = 'center';
                    setTimeout(function() {
                        toast.style.opacity = '1';
                        toast.style.transform = 'translateY(0)';
                    }, 50);
                    toast.onclick = function() {
                        toast.style.opacity = '0';
                        toast.style.transform = 'translateY(10px)';
                        setTimeout(function() { toast.style.display = 'none'; }, 500);
                        var chatBtn = document.getElementById('chat-toggle-btn');
                        if (chatBtn) chatBtn.click();
                    };
                    setTimeout(function() {
                        if (toast.style.opacity !== '0') {
                            toast.style.opacity = '0';
                            toast.style.transform = 'translateY(10px)';
                            setTimeout(function() { toast.style.display = 'none'; }, 500);
                        }
                    }, 8000);
                }, 700);
            }, 800);
        }

        /* Disable animation-timer — we're done polling */
        return [window.dash_clientside.no_update, true];
    }
    """,
    [Output('fade-complete-timer', 'disabled'),
     Output('animation-timer', 'disabled', allow_duplicate=True)],
    Input('animation-timer', 'n_intervals'),
    prevent_initial_call=True
)

# Hide overlay and toast when navigating back to selection screen
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            var overlay = document.getElementById('plot-loading-overlay');
            var toast = document.getElementById('ai-hint-toast');
            if (overlay) { overlay.style.display = 'none'; overlay.style.opacity = '0'; }
            if (toast) { toast.style.display = 'none'; toast.style.opacity = '0'; }
        }
        return '';
    }
    """,
    Output('back-button', 'title'),
    Input('back-button', 'n_clicks'),
    prevent_initial_call=True
)


_APP_CONTEXT = """
You are running inside the **Marine Mammal Proteome Explorer** — an interactive UMAP visualization tool.

How the app works:
- Each dot on the UMAP plot is a single protein, coloured by species.
- Proteins are grouped into **clusters** using HDBSCAN run on the 2D UMAP coordinates. Proteins that cluster together share similar predicted structure/function based on ProtT5-XL embeddings.
- Cluster -1 means noise (unclustered proteins).
- **Cluster hull colours reflect cluster size** (number of proteins in the cluster): the colour is a continuous gradient — blue = smallest clusters, cyan → green → yellow → red = largest clusters. The exact size of a cluster is shown when you select it. Do NOT guess a cluster's size from its colour alone — always check the actual protein count shown after selecting it.
- **Clicking a cluster centroid label** (or its hull border) selects it: proteins are highlighted in gold, and the AI analyses the cluster's enriched domains.
- **Domain enrichment** is computed with a hypergeometric test: a domain is enriched if it appears far more often in a cluster than expected given its frequency across the whole proteome (fold > 1.5×, p < 0.05).
- The AI can highlight proteins on the plot, search for specific proteins or domains, compare species, and answer biology questions grounded in marine mammal research literature (via RAG over an embedded literature database).
- Users can control the plot by asking the AI (show unique/shared proteins, add/remove species, resize points, change opacity).
""".strip()

_QUESTION_PREFIX_RE = re.compile(
    r'^(what\s+is|what\s+are|what\s+does|tell\s+me\s+about|where\s+is|'
    r'explain|describe|how\s+does|why\s+is|find|locate|show\s+me|'
    r'search\s+for|look\s+up|give\s+me\s+info\s+on|do\s+you\s+know\s+about)\s+',
    re.IGNORECASE,
)

_ENRICHMENT_CACHE = {}   # frozenset(species_names) -> precomputed background stats


def compute_cluster_enrichment(active_data, cluster_id=None, top_n=8):
    """Compute hypergeometric domain enrichment for clusters in active_data.

    If cluster_id is given, returns enrichment for that specific cluster only.
    Otherwise returns a dict mapping cluster -> list of (domain, fold, pval).

    Uses the hypergeometric test: given a domain that appears K times across
    total_proteins, what is the probability of seeing >= k copies in a cluster
    of size cl_size purely by chance?  fold > 1.5x AND p < 0.05 to be included.
    """
    cache_key = frozenset(sd['name'] for sd in active_data)
    if cache_key in _ENRICHMENT_CACHE:
        cluster_species, cluster_domains, cluster_protein_names, \
            background_domains, domain_descriptions, total_proteins = _ENRICHMENT_CACHE[cache_key]
    else:
        cluster_species = {}
        cluster_domains = {}
        cluster_protein_names = {}
        background_domains = {}
        domain_descriptions = {}
        total_proteins = 0

        for sd in active_data:
            display = SPECIES_DATA.get(sd['name'], {}).get('display_name', sd['name'])
            for rec in sd.get('df', []):
                cl = rec.get('Cluster Label')
                if cl is None or str(cl) == 'nan':
                    continue
                cl = int(cl)
                if cl < 0:
                    continue
                entry_id = str(rec.get('Entry', '') or '')
                total_proteins += 1

                cluster_species.setdefault(cl, {})
                cluster_species[cl][display] = cluster_species[cl].get(display, 0) + 1

                ann = ANNOTATIONS.get(entry_id, {})
                labels = (ann.get('hmm_labels', '') or '').split(';')
                descs  = (ann.get('hmm_descriptions', '') or '').split(';')
                for i, dom in enumerate(labels):
                    dom = dom.strip()
                    if not dom:
                        continue
                    cluster_domains.setdefault(cl, {})
                    cluster_domains[cl][dom] = cluster_domains[cl].get(dom, 0) + 1
                    background_domains[dom] = background_domains.get(dom, 0) + 1
                    if dom not in domain_descriptions and i < len(descs):
                        desc = descs[i].strip()
                        if desc:
                            domain_descriptions[dom] = desc

                pname = (ann.get('desc', '') or '').split(' OS=')[0].strip()[:80]
                if pname:
                    cluster_protein_names.setdefault(cl, set()).add(pname)

        _ENRICHMENT_CACHE[cache_key] = (
            cluster_species, cluster_domains, cluster_protein_names,
            background_domains, domain_descriptions, total_proteins
        )

    def _enriched(cl):
        cl_size = sum(cluster_species.get(cl, {}).values())
        if cl_size == 0:
            return []
        results = []
        for dom, k in cluster_domains.get(cl, {}).items():
            K = background_domains.get(dom, 0)
            fold = (k / cl_size) / (K / total_proteins) if K and total_proteins else 0
            if fold <= 1.5:
                continue
            pval = hypergeom.sf(k - 1, total_proteins, K, cl_size)
            if pval < 0.05:
                results.append((dom, fold, pval))
        results.sort(key=lambda x: x[2])
        return results[:top_n]

    if cluster_id is not None:
        sp_counts = cluster_species.get(cluster_id, {})
        sharing = ("unique to " + next(iter(sp_counts))) if len(sp_counts) == 1 else ("shared: " + ", ".join(sp_counts))
        size = sum(sp_counts.values())
        enriched = _enriched(cluster_id)
        return {
            'cluster': cluster_id,
            'size': size,
            'sharing': sharing,
            'enriched_domains': enriched,
            'sample_names': sorted(cluster_protein_names.get(cluster_id, set()))[:25],
            'domain_descriptions': {dom: domain_descriptions.get(dom, '') for dom, _, _ in enriched},
        }

    return {cl: _enriched(cl) for cl in cluster_species}


def _interpret_cluster(info, active_species):
    """Ask GPT to interpret cluster biology. Uses RAG to ground the answer in literature.

    Returns a markdown string suitable for the chat panel.
    """
    cl_id    = info['cluster']
    size     = info['size']
    sharing  = info['sharing']
    enriched = info.get('enriched_domains', [])
    names    = info.get('sample_names', [])
    descs    = info.get('domain_descriptions', {})

    # Build the structured cluster summary for the prompt
    domain_lines = []
    for dom, fold, pval in enriched:
        long_name = descs.get(dom, '')
        label = f"{dom} ({long_name})" if long_name else dom
        domain_lines.append(f"  - {label}: {fold:.1f}× enriched, p={pval:.1e}")

    proteins_str = "; ".join(names) if names else "—"
    cluster_summary = (
        f"Cluster {cl_id}: {size} proteins | {sharing}\n\n"
        f"Statistically enriched Pfam domains:\n"
        + ("\n".join(domain_lines) if domain_lines else "  (none above threshold)")
        + f"\n\nExample proteins:\n  {proteins_str}"
    )

    # RAG: pull literature about the top enriched domains / protein names
    rag_block = ""
    rag_docs = []
    rag_terms = [descs.get(d, d) for d, _, _ in enriched[:3]] + names[:3]
    rag_query = " ".join(t for t in rag_terms if t) + " marine mammal function biology"
    try:
        vs = _load_vectorstore()
        rag_docs = vs.similarity_search(
            rag_query, k=5,
            filter={"content_type": {"$eq": "text"}},
        )
        if rag_docs:
            rag_block = "\n\n---\n\n".join(d.page_content for d in rag_docs)
    except Exception as e:
        print(f"[_interpret_cluster] RAG error: {e}")

    system_msg = (
        "You are a molecular biologist interpreting a protein cluster from a marine mammal proteome UMAP.\n\n"
        "You are given:\n"
        "- Pfam domains that are STATISTICALLY ENRICHED in this cluster (hypergeometric test, fold > 1.5×)\n"
        "- The actual names of proteins found in this cluster\n"
        "- Which species the cluster is shared between\n"
        "- Optionally: literature context from marine mammal research papers\n\n"
        "YOUR JOB — be specific and concrete:\n\n"
        "**What are these proteins?**\n"
        "Name the specific proteins listed. Don't say 'various proteins' — say what they actually are "
        "(e.g. 'zinc finger protein 160-like, zinc finger protein 852'). "
        "State the protein family they belong to.\n\n"
        "**What do the enriched domains do — exactly?**\n"
        "For each enriched domain, explain its SPECIFIC molecular function: "
        "What does it bind? What reaction does it catalyse? What structure does it form? "
        "For example, don't say 'involved in gene regulation' — say 'binds specific DNA sequences "
        "via coordinated zinc ions, recruiting transcriptional repressors to silence target genes'. "
        "Include the fold-enrichment to convey how strongly enriched it is.\n\n"
        "**What is this cluster's biological role?**\n"
        "Based on the specific proteins and domains, what cellular process does this cluster serve? "
        "If the literature context mentions these proteins or domains in the context of these species, "
        "cite that directly. If the literature has nothing specific, say so explicitly: "
        "'The literature database does not contain specific findings for [domain] in [species] — "
        "based on general biology, this domain is known to...'\n\n"
        "RULES:\n"
        "- Before submitting your answer, scan it for the words: might, may, could, likely, possibly, "
        "suggest, perhaps, probably. Every time you find one, either (a) replace it with a direct "
        "statement backed by the data/literature provided, or (b) rewrite as: 'The literature database "
        "contains no specific evidence for this in [species] — based on general molecular biology, "
        "[domain] is known to [specific function].' Do not leave any speculative language in the output.\n"
        "- If a domain is very common (e.g. zinc finger C2H2 with hundreds of members in mammalian "
        "genomes), say so explicitly: 'This is a highly abundant domain family — enrichment here "
        "indicates these proteins share structural similarity but does not pinpoint a specific function "
        "without knowing their target genes.'\n"
        "- Never say 'more research is needed'.\n"
        "- Never generalise when you have specific protein names in front of you — use them.\n"
        "- If fold enrichment is very high (>5×), highlight that as unusual and worth investigating.\n"
        "- Max 400 words."
    )

    user_content = cluster_summary
    if rag_block:
        user_content = f"Literature context from marine mammal papers:\n{rag_block}\n\n---\n\nCluster data:\n{cluster_summary}"

    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        interpretation = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[_interpret_cluster] GPT error: {e}")
        dom_str = ", ".join(f"{d} ({fold:.1f}×)" for d, fold, _ in enriched) if enriched else "none detected"
        interpretation = (
            f"**Enriched domains:** {dom_str}\n\n"
            f"**Example proteins:** {proteins_str}"
        )

    sources = _build_sources_block(rag_docs) if rag_docs else ""
    header = f"**Cluster {cl_id}** — {size} proteins, {sharing}\n\n"
    footer = f"\n\n{sources}" if sources else ""
    return header + interpretation + footer


def _extract_cluster_ids_from_text(text):
    """Return sorted list of unique cluster numbers mentioned in text.

    Handles:
      'Cluster 42'                 → [42]
      'Clusters 5 and 12'          → [5, 12]
      'Clusters 3, 7, and 15'      → [3, 7, 15]
      'Cluster 3, Cluster 7'       → [3, 7]
      'cluster 5/12'               → [5, 12]
    """
    import re as _re
    ids = set()
    # Match "cluster(s) N" — handles single numbers
    for m in _re.finditer(r'\bclusters?\s+(\d+)', text, _re.IGNORECASE):
        ids.add(int(m.group(1)))
    # Match "cluster(s) N, M, and P" — grab all numbers in a comma/and/slash list
    for m in _re.finditer(
        r'\bclusters?\s+(\d+(?:\s*[,/&]\s*(?:and\s+)?\d+|\s+and\s+\d+)*)',
        text, _re.IGNORECASE
    ):
        for num in _re.findall(r'\d+', m.group(1)):
            ids.add(int(num))
    return sorted(ids)


def _merge_highlight_for_clusters(cluster_ids, loaded_data, active_species):
    """Merge highlight data for multiple cluster ids into one highlight-store dict."""
    xs, ys, entry_ids = [], [], []
    for cl_id in cluster_ids:
        result = _resolve_highlight_query(str(cl_id), loaded_data, active_species)
        if result:
            xs.extend(result['xs'])
            ys.extend(result['ys'])
            entry_ids.extend(result['entry_ids'])
    if not xs:
        return None
    label = (
        f"Clusters {', '.join(str(c) for c in cluster_ids)}"
        if len(cluster_ids) > 1 else f"Cluster {cluster_ids[0]}"
    )
    return {"xs": xs, "ys": ys, "label": label, "entry_ids": entry_ids}


def _cluster_search(term, loaded_data, active_species):
    """Find which clusters contain proteins matching term (name or domain).

    Returns (answer_text, highlight_data).
    """
    if not loaded_data or not term:
        return "No data loaded.", None

    active_data = [sd for sd in loaded_data if sd['name'] in (active_species or [])]
    term_lower = term.lower()

    # Map cluster -> {species -> count} and cluster -> set of protein names
    cluster_hits = {}   # cl -> {display_name -> count}
    cluster_names = {}  # cl -> [protein names]
    unclustered_hits = {}        # display_name -> [protein names] for cl == -1 matches
    unclustered_coords = []      # (species_name, umap1, umap2) for noise matches — used for nearest-cluster lookup
    # Also collect cluster centroid data per species from the full data
    species_cluster_pts = {}     # species_name -> {cl -> [(umap1, umap2)]}

    for sd in active_data:
        display = SPECIES_DATA.get(sd['name'], {}).get('display_name', sd['name'])
        # Use full unsampled data so rare proteins (e.g. myoglobin) are not missed
        search_records = _FULL_SEARCH_DATA.get(sd['name'], sd.get('df', []))
        # Collect cluster points for this species (for nearest-cluster lookup)
        cl_pts = {}
        for rec in search_records:
            cl_r = rec.get('Cluster Label')
            if cl_r is None or str(cl_r) == 'nan':
                continue
            cl_r = int(cl_r)
            if cl_r < 0:
                continue
            x, y = rec.get('UMAP 1'), rec.get('UMAP 2')
            if x is not None and y is not None:
                cl_pts.setdefault(cl_r, []).append((x, y))
        species_cluster_pts[sd['name']] = cl_pts

        for rec in search_records:
            cl = rec.get('Cluster Label')
            if cl is None or str(cl) == 'nan':
                continue
            cl = int(cl)
            entry_id = str(rec.get('Entry', '') or '')
            # Search CSV protein name directly (most reliable — always present)
            csv_name = str(rec.get('Protein names', '') or '').lower()
            ann = ANNOTATIONS.get(entry_id, {})
            desc_lower  = (ann.get('desc', '') or '').split(' OS=')[0].lower()
            hmm_lower   = (ann.get('hmm_labels', '') or '').lower()
            hmm_d_lower = (ann.get('hmm_descriptions', '') or '').lower()
            if (term_lower in csv_name or term_lower in desc_lower
                    or term_lower in hmm_lower or term_lower in hmm_d_lower):
                pname = (
                    (ann.get('desc', '') or '').split(' OS=')[0].strip()[:60]
                    or str(rec.get('Protein names', '') or '').split('(')[0].strip()[:60]
                )
                if cl < 0:
                    # Track unclustered (noise) matches separately
                    unclustered_hits.setdefault(display, [])
                    if pname and pname not in unclustered_hits[display]:
                        unclustered_hits[display].append(pname)
                    x, y = rec.get('UMAP 1'), rec.get('UMAP 2')
                    if x is not None and y is not None:
                        unclustered_coords.append((sd['name'], x, y))
                    continue
                cluster_hits.setdefault(cl, {})
                cluster_hits[cl][display] = cluster_hits[cl].get(display, 0) + 1
                if pname:
                    cluster_names.setdefault(cl, [])
                    if pname not in cluster_names[cl]:
                        cluster_names[cl].append(pname)

    if not cluster_hits:
        if unclustered_hits:
            # Protein exists but HDBSCAN classified it as noise.
            # Find the nearest assigned cluster for each match to give spatial context.
            nearest_info = []
            seen_species = set()
            for sp_name, nx, ny in unclustered_coords:
                if sp_name in seen_species:
                    continue
                cl_pts = species_cluster_pts.get(sp_name, {})
                if not cl_pts:
                    continue
                centroid_means = {
                    cl: (np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]))
                    for cl, pts in cl_pts.items()
                }
                best_cl = min(centroid_means, key=lambda cl: (centroid_means[cl][0]-nx)**2 + (centroid_means[cl][1]-ny)**2)
                best_d = ((centroid_means[best_cl][0]-nx)**2 + (centroid_means[best_cl][1]-ny)**2)**0.5
                display = SPECIES_DATA.get(sp_name, {}).get('display_name', sp_name)
                nearest_info.append((display, best_cl, round(best_d, 1)))
                seen_species.add(sp_name)

            sp_parts = list(unclustered_hits.keys())
            nearest_str = ''
            if nearest_info:
                nearest_str = ' Nearest cluster(s): ' + ', '.join(
                    f'**Cluster {cl}** ({sp}, UMAP distance {d:.1f})' for sp, cl, d in nearest_info[:3]
                ) + '.'

            # Count total unclustered matches across all species
            total_unc = sum(len(v) for v in unclustered_hits.values())
            return (
                f'**"{term}"** exists in the dataset ({", ".join(sp_parts)}, '
                f'{total_unc} matching protein(s)) but is not assigned to any cluster. '
                f'HDBSCAN classified it as noise because this protein family is too small '
                f'to meet the minimum cluster size threshold (150 proteins). '
                f'It is tightly co-localised in UMAP space with related proteins but the group '
                f'is simply too small to form a formal cluster.'
                f'{nearest_str}',
                None
            )
        return (
            f'No proteins matching **"{term}"** found in the currently active species. '
            'Try a different search term or add more species to the view.', None
        )

    # Sort by total hit count descending
    sorted_clusters = sorted(cluster_hits.items(), key=lambda x: -sum(x[1].values()))

    lines = [f'Found **"{term}"** in **{len(cluster_hits)} cluster(s)**:\n']
    for cl, sp_counts in sorted_clusters[:20]:
        total = sum(sp_counts.values())
        sp_str = ', '.join(f'{sp} ({n})' for sp, n in sp_counts.items())
        names_preview = '; '.join(cluster_names.get(cl, [])[:2])
        lines.append(f'- **Cluster {cl}** — {total} protein(s) | {sp_str}' +
                     (f' | e.g. {names_preview}' if names_preview else ''))
    if len(sorted_clusters) > 20:
        lines.append(f'… and {len(sorted_clusters) - 20} more clusters.')

    answer = '\n'.join(lines)
    cluster_ids = [cl for cl, _ in sorted_clusters]
    highlight = _merge_highlight_for_clusters(cluster_ids, loaded_data, active_species)
    return answer, highlight


_GENERIC_BIOLOGY_TERMS = {
    'protein', 'proteins', 'proteome', 'proteomes', 'enzyme', 'enzymes',
    'gene', 'genes', 'cluster', 'clusters', 'species', 'organism', 'organisms',
    'domain', 'domains', 'sequence', 'sequences', 'amino acid', 'amino acids',
    'peptide', 'peptides', 'cell', 'cells', 'molecule', 'molecules',
    'function', 'functions', 'structure', 'structures', 'biology', 'marine',
    'mammal', 'mammals', 'whale', 'seal', 'dolphin', 'orca', 'hippo', 'bear',
}

def _extract_protein_search_term(query):
    """Strip leading question words and return the core search term.
    Returns None if the term is too short or a generic biology word."""
    cleaned = _QUESTION_PREFIX_RE.sub('', query.strip()).rstrip('?').strip()
    if len(cleaned) < 4:
        return None
    if cleaned.lower() in _GENERIC_BIOLOGY_TERMS:
        return None
    return cleaned


def _dataset_lookup_context(query, loaded_data, active_species):
    """Search dataset for a protein name and return a formatted context string, or ''."""
    term = _extract_protein_search_term(query)
    if not term:
        return ''
    matches = _resolve_highlight_query(term, loaded_data, active_species)
    if not matches or not matches.get('entry_ids'):
        return ''
    # If we matched more than 200 proteins the term is too generic — skip
    if len(matches['entry_ids']) > 200:
        return ''
    rows = _build_protein_rows(matches['entry_ids'], loaded_data, active_species)
    if not rows:
        return ''
    # Group by cluster then species
    cluster_map = {}
    for r in rows:
        cl = r['Cluster']
        sp = r['Organism']
        cluster_map.setdefault(cl, {}).setdefault(sp, 0)
        cluster_map[cl][sp] += 1
    lines = [f'Dataset lookup for "{term}" ({len(rows)} match{"es" if len(rows) != 1 else ""}):']
    for cl, species_counts in sorted(cluster_map.items(), key=lambda x: x[0]):
        sp_str = ', '.join(f'{sp} ({n})' for sp, n in species_counts.items())
        lines.append(f'  Cluster {cl}: {sp_str}')
    # Sample a few protein names for context
    sample_names = list({r['Protein Name'] for r in rows if r['Protein Name'] != '—'})[:3]
    if sample_names:
        lines.append(f'  Example names: {"; ".join(sample_names)}')
    return '\n'.join(lines)


def _resolve_highlight_query(query, loaded_data, active_species):
    """Return highlight-store data dict for a cluster number, Entry ID, or protein name query."""
    if not loaded_data or not query:
        return None
    query = query.strip()
    active_species = active_species or []
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    if not active_data:
        return None

    xs, ys, label = [], [], ""
    entry_ids = []

    # 1. Try cluster number
    try:
        cluster_num = int(query.lstrip('#').strip())
        for sd in active_data:
            for rec in sd['df']:
                cl = rec.get('Cluster Label')
                if cl is not None and str(cl) != 'nan' and int(cl) >= 0 and int(cl) == cluster_num:
                    xs.append(rec['UMAP 1 Scaled'])
                    ys.append(rec['UMAP 2 Scaled'])
                    entry_ids.append((sd['name'], str(rec.get('Entry', ''))))
        if xs:
            label = f"Cluster {cluster_num}"
        return {"xs": xs, "ys": ys, "label": label, "entry_ids": entry_ids} if xs else None
    except ValueError:
        pass

    # 2. Try exact/prefix Entry ID match
    query_up = query.upper()
    for sd in active_data:
        for rec in sd['df']:
            entry = str(rec.get('Entry', '') or '').upper()
            if entry == query_up or entry.startswith(query_up):
                xs.append(rec['UMAP 1 Scaled'])
                ys.append(rec['UMAP 2 Scaled'])
                entry_ids.append((sd['name'], str(rec.get('Entry', ''))))
                label = str(rec.get('Entry', query))
    if xs:
        return {"xs": xs, "ys": ys, "label": label, "entry_ids": entry_ids}

    # 3. Protein name search — match against CSV 'Protein names' and ANNOTATIONS desc.
    # Use full unsampled data so rare proteins (e.g. myoglobin) are not missed.
    # We need scaled UMAP coords for highlighting, so build a lookup from Entry -> scaled coords
    # from the display df, then fall back to raw UMAP coords scaled on-the-fly if not in sample.
    query_lower = query.lower()

    # Build scale factors from the display df (already has 'UMAP 1 Scaled')
    scale_lookup = {}  # (species_name, entry) -> (x_scaled, y_scaled)
    for sd in active_data:
        for rec in sd['df']:
            e = str(rec.get('Entry', '') or '')
            if 'UMAP 1 Scaled' in rec and rec['UMAP 1 Scaled'] is not None:
                scale_lookup[(sd['name'], e)] = (rec['UMAP 1 Scaled'], rec['UMAP 2 Scaled'])

    # Derive per-species linear scale/offset from sampled display records (least-squares fit).
    # The transformation raw -> scaled is always linear, so any two+ points give an exact fit.
    species_scale = {}  # species_name -> (a1, b1, a2, b2)
    for sd in active_data:
        recs = sd['df']
        pairs1 = [(r['UMAP 1'], r['UMAP 1 Scaled']) for r in recs
                  if r.get('UMAP 1') is not None and r.get('UMAP 1 Scaled') is not None]
        pairs2 = [(r['UMAP 2'], r['UMAP 2 Scaled']) for r in recs
                  if r.get('UMAP 2') is not None and r.get('UMAP 2 Scaled') is not None]
        if len(pairs1) >= 2:
            raw1, sc1 = zip(*pairs1); raw2, sc2 = zip(*pairs2)
            a1, b1 = np.polyfit(raw1, sc1, 1)
            a2, b2 = np.polyfit(raw2, sc2, 1)
            species_scale[sd['name']] = (a1, b1, a2, b2)

    for sd in active_data:
        search_records = _FULL_SEARCH_DATA.get(sd['name'], sd.get('df', []))
        for rec in search_records:
            entry_id = str(rec.get('Entry', '') or '')
            csv_name = str(rec.get('Protein names', '') or '').lower()
            ann = ANNOTATIONS.get(entry_id, {})
            ann_name = (ann.get('desc', '') or '').split(' OS=')[0].lower()
            if query_lower not in csv_name and query_lower not in ann_name:
                continue
            # Get scaled coords — from display sample if available, else derive via linear map
            key = (sd['name'], entry_id)
            if key in scale_lookup:
                x_sc, y_sc = scale_lookup[key]
            elif sd['name'] in species_scale:
                a1, b1, a2, b2 = species_scale[sd['name']]
                raw_x = rec.get('UMAP 1')
                raw_y = rec.get('UMAP 2')
                if raw_x is None or raw_y is None:
                    continue
                x_sc = a1 * raw_x + b1
                y_sc = a2 * raw_y + b2
            else:
                continue
            xs.append(x_sc)
            ys.append(y_sc)
            entry_ids.append((sd['name'], entry_id))
    if xs:
        label = f'"{query}"'
        return {"xs": xs, "ys": ys, "label": label, "entry_ids": entry_ids}

    return None


def _resolve_ui_command(text, current_size, current_opacity):
    """Parse a CONTROL: command and return (filter_val, size_val, opacity_val, reply).

    Any value that shouldn't change is returned as dash.no_update.
    """
    t = text.lower().strip()
    no = dash.no_update

    # --- Protein filter ---
    if any(w in t for w in ['unique', 'species-specific', 'species specific']):
        return 'unique', no, no, "Showing only **species-specific** proteins — those not grouped into any shared functional cluster. 🔬"
    if 'shared' in t or 'common' in t or 'conserved' in t:
        return 'shared', no, no, "Showing only **shared** proteins — those found across multiple species. 🌊"
    if any(p in t for p in ['all protein', 'show all', 'reset filter', 'clear filter', 'no filter']):
        return 'all', no, no, "Showing **all proteins** — no filter applied."

    # --- Point size ---
    import re as _re
    size_match = _re.search(r'\b(?:size|point size|dot size)\s*(?:to\s*)?(\d+)', t)
    if size_match:
        val = max(1, min(10, int(size_match.group(1))))
        return no, val, no, f"Point size set to **{val}**."
    if any(p in t for p in ['bigger', 'larger', 'increase size', 'make larger', 'size up', 'zoom in on dot', 'enlarge']):
        val = min(10, (current_size or 3) + 2)
        return no, val, no, f"Made the points bigger (size {val})."
    if any(p in t for p in ['smaller', 'tinier', 'decrease size', 'make smaller', 'size down', 'shrink']):
        val = max(1, (current_size or 3) - 2)
        return no, val, no, f"Made the points smaller (size {val})."

    # --- Opacity ---
    opacity_match = _re.search(r'\bopacity\s*(?:to\s*)?(0?\.\d+|\d+(?:\.\d+)?)', t)
    if opacity_match:
        val = max(0.1, min(1.0, float(opacity_match.group(1))))
        return no, no, val, f"Opacity set to **{val:.1f}**."
    if any(p in t for p in ['more transparent', 'less opaque', 'fade', 'lower opacity', 'decrease opacity']):
        val = max(0.1, round((current_opacity or 1.0) - 0.2, 1))
        return no, no, val, f"Opacity reduced to **{val:.1f}**."
    if any(p in t for p in ['more opaque', 'less transparent', 'brighter', 'higher opacity', 'increase opacity', 'full opacity']):
        val = min(1.0, round((current_opacity or 1.0) + 0.2, 1))
        return no, no, val, f"Opacity increased to **{val:.1f}**."

    # --- Plotly toolbar things (guide user) ---
    if any(p in t for p in ['zoom', 'pan', 'reset', 'autoscale', 'download', 'screenshot', 'save image']):
        return no, no, no, (
            "For zooming, panning, and resetting the view, use the **toolbar in the top-right corner of the plot** — "
            "it has zoom in/out, pan, lasso, and reset axes buttons. To save a screenshot, click the camera icon there."
        )

    return no, no, no, (
        f"I didn't quite understand that control command: *{text}*. "
        "You can say things like: *'show unique proteins'*, *'show shared proteins'*, "
        "*'make the points bigger'*, *'set opacity to 0.5'*."
    )


def _resolve_species_command(text, active_species):
    """Parse a SPECIES: command and return (new_active_species, reply_message).

    Supported actions (case-insensitive in text):
      add <name>        — add species to active list
      remove <name>     — remove species from active list
      show only <name>  — show only that species
      add all           — show all species
      remove all / hide all — keep only one (prevent empty list)
    Returns (new_list, reply) or (None, error_message).
    """
    active_species = list(active_species or [])
    all_species = list(SPECIES_DATA.keys())

    def fuzzy_match(name_query):
        """Return SPECIES_DATA key for best match, or None."""
        q = name_query.lower().strip()
        # Exact key match
        for k in all_species:
            if k.lower() == q:
                return k
        # Substring match on display_name or key
        candidates = []
        for k in all_species:
            display = SPECIES_DATA[k]['display_name'].lower()
            key = SPECIES_DATA[k]['key'].lower()
            if q in display or q in key or display in q:
                candidates.append(k)
        return candidates[0] if len(candidates) == 1 else (candidates[0] if candidates else None)

    text = text.strip()
    text_lower = text.lower()

    # "add all" / "show all"
    if any(p in text_lower for p in ['add all', 'show all', 'all species', 'every species']):
        return all_species, f"Showing all species on the plot! 🌊"

    # "show only <name>"
    for prefix in ['show only ', 'only show ', 'isolate ']:
        if text_lower.startswith(prefix):
            name_q = text[len(prefix):].strip()
            match = fuzzy_match(name_q)
            if match:
                display = SPECIES_DATA[match]['display_name']
                return [match], f"Isolating **{display}** on the plot."
            return None, f"I couldn't find a species called **{name_q}**. Available: {', '.join(SPECIES_DATA[k]['display_name'] for k in all_species)}."

    # "remove <name>" / "hide <name>"
    for prefix in ['remove ', 'hide ', 'take off ', 'turn off ']:
        if text_lower.startswith(prefix):
            name_q = text[len(prefix):].strip()
            match = fuzzy_match(name_q)
            if match:
                display = SPECIES_DATA[match]['display_name']
                if match not in active_species:
                    return None, f"**{display}** is already hidden."
                if len(active_species) <= 1:
                    return None, f"Can't remove **{display}** — you need at least one species visible."
                new_list = [s for s in active_species if s != match]
                return new_list, f"Removed **{display}** from the plot."
            return None, f"I couldn't find a species called **{name_q}**."

    # "add <name>" / "compare to <name>" / "compare with <name>"
    for prefix in ['add ', 'compare to ', 'compare with ', 'add in ', 'also show ', 'include ']:
        if text_lower.startswith(prefix):
            name_q = text[len(prefix):].strip()
            match = fuzzy_match(name_q)
            if match:
                display = SPECIES_DATA[match]['display_name']
                if match in active_species:
                    return None, f"**{display}** is already visible on the plot."
                return active_species + [match], f"Added **{display}** to the plot for comparison! 🔬"
            return None, f"I couldn't find a species called **{name_q}**. Available: {', '.join(SPECIES_DATA[k]['display_name'] for k in all_species)}."

    # Fallback: try to fuzzy match the whole text as a species name with "add"
    match = fuzzy_match(text)
    if match:
        display = SPECIES_DATA[match]['display_name']
        if match in active_species:
            return None, f"**{display}** is already visible on the plot."
        return active_species + [match], f"Added **{display}** to the plot! 🔬"

    return None, f"I'm not sure what to do with **{text}** — try something like 'add harbor seal', 'remove orca', or 'show only gray whale'."


# --- Highlight search callback (UI search bar) ---
@app.callback(
    Output('highlight-store', 'data'),
    [Input('highlight-search-btn', 'n_clicks'),
     Input('highlight-clear-btn', 'n_clicks'),
     Input('highlight-search-input', 'n_submit')],
    [State('highlight-search-input', 'value'),
     State('loaded-data', 'data'),
     State('active-species-store', 'data')],
    prevent_initial_call=True,
)
def ui_highlight_search(search_clicks, clear_clicks, n_submit, search_value, loaded_data, active_species):
    if ctx.triggered_id == 'highlight-clear-btn':
        return None
    if not search_value:
        return None
    terms = [t.strip() for t in search_value.split(',') if t.strip()]
    if len(terms) <= 1:
        return _resolve_highlight_query(search_value, loaded_data, active_species)
    # Merge results from all terms
    merged_xs, merged_ys, merged_ids = [], [], []
    for term in terms:
        result = _resolve_highlight_query(term, loaded_data, active_species)
        if result and result.get('xs'):
            merged_xs.extend(result['xs'])
            merged_ys.extend(result['ys'])
            merged_ids.extend(result.get('entry_ids', []))
    if not merged_xs:
        return None
    return {'xs': merged_xs, 'ys': merged_ys, 'entry_ids': merged_ids, 'label': search_value}


# --- Highlight protein when row is double-clicked in the protein table ---
# Single click expands the row via CSS (state: active) without changing the UMAP.
# Double click (clicking the same row twice) updates highlight-store.
@app.callback(
    Output('last-click-store', 'data'),
    Output('highlight-store', 'data', allow_duplicate=True),
    Input('protein-datatable', 'active_cell'),
    [State('protein-datatable', 'data'),
     State('loaded-data', 'data'),
     State('active-species-store', 'data'),
     State('last-click-store', 'data')],
    prevent_initial_call=True,
)
def table_row_click(active_cell, table_data, loaded_data, active_species, last_click):
    if not active_cell or not table_data:
        raise dash.exceptions.PreventUpdate
    row_idx = active_cell['row']
    row = table_data[row_idx]
    entry_id = row.get('Entry', '')
    is_double = (
        last_click is not None and
        last_click.get('row') == row_idx and
        last_click.get('entry') == entry_id
    )
    if is_double:
        # Double click: highlight in UMAP and reset so the next single click on
        # the same row doesn't re-trigger.
        result = _resolve_highlight_query(entry_id, loaded_data, active_species)
        if not result or not result.get('entry_ids'):
            protein_name = row.get('Protein Name', '')
            result = _resolve_highlight_query(protein_name, loaded_data, active_species)
        return None, result
    else:
        # Single click: just track which row was clicked, leave UMAP alone.
        return {'row': row_idx, 'entry': entry_id}, dash.no_update


# --- Toggle species in active-species-store ---
@app.callback(
    Output('active-species-store', 'data', allow_duplicate=True),
    [Input(f'toggle-{species}', 'n_clicks') for species in SPECIES_DATA] +
    [Input('loaded-data', 'data')],
    [State('active-species-store', 'data'),
     State('selected-species-store', 'data')],
    prevent_initial_call=True
)
def update_active_species(*args):
    species_list = list(SPECIES_DATA.keys())
    n_species = len(species_list)
    # args: n_clicks for each toggle (n_species), loaded_data (1), active_species state (1), selected_species state (1)
    loaded_data = args[n_species]
    active_species = args[n_species + 1] or []
    selected_species = args[n_species + 2] or []

    trigger = ctx.triggered_id
    if trigger is None:
        raise dash.exceptions.PreventUpdate

    # When loaded-data fires, initialize from selected-species-store
    if trigger == 'loaded-data':
        return selected_species if selected_species else list(SPECIES_DATA.keys())

    # A toggle button was clicked
    if isinstance(trigger, str) and trigger.startswith('toggle-'):
        clicked_species = trigger.replace('toggle-', '')
        # Need loaded data to be present
        if not loaded_data:
            raise dash.exceptions.PreventUpdate
        if clicked_species in active_species:
            # Don't allow removing the last species
            if len(active_species) <= 1:
                raise dash.exceptions.PreventUpdate
            return [s for s in active_species if s != clicked_species]
        else:
            return active_species + [clicked_species]

    raise dash.exceptions.PreventUpdate


# --- Style toggle buttons based on active species ---
@app.callback(
    [Output(f'toggle-{species}', 'style') for species in SPECIES_DATA],
    [Input('active-species-store', 'data')]
)
def update_toggle_styles(active_species):
    active_species = active_species or []
    styles = []
    for species in SPECIES_DATA:
        species_color = SPECIES_DATA[species]['color']
        is_active = species in active_species
        style = {
            'display': 'inline-flex', 'alignItems': 'center', 'gap': '4px',
            'padding': '3px 8px',
            'background': 'rgba(255, 255, 255, 0.05)' if is_active else 'rgba(255, 255, 255, 0.02)',
            'borderRadius': 8,
            'border': f'1px solid {species_color}' if is_active else '1px solid rgba(255,255,255,0.15)',
            'cursor': 'pointer',
            'opacity': 1.0 if is_active else 0.4,
            'color': species_color if is_active else '#a8dadc',
            'transition': 'all 0.2s ease',
        }
        styles.append(style)
    return styles

@app.callback(
    Output('interactive-graph', 'figure'),
    [Input('point-size', 'value'),
     Input('opacity', 'value'),
     Input('loaded-data', 'data'),
     Input('protein-filter', 'value'),
     Input('active-species-store', 'data'),
     Input('highlight-store', 'data')]
)
def update_interactive_graph(point_size, opacity, loaded_data, protein_filter, active_species, highlight_data):
    if not loaded_data:
        return go.Figure()
    active_species = active_species or []
    # Filter to only active species
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    if not active_data:
        return go.Figure()
    # Recompute filter tags dynamically for the active subset
    _compute_filter_tags(active_data)

    fig = go.Figure()

    # Build convex hulls for all clusters, merged into 2 traces (fill + border) using None
    # separators. This replaces 263 individual traces with 2, dramatically reducing browser load.
    # Skip entirely when species-specific filter is active (no clusters shown then).
    cluster_points = {}
    if protein_filter != 'unique':
        for sd in active_data:
            for r in sd['df']:
                cl = r.get('Cluster Label')
                if cl is None or str(cl) == 'nan' or int(cl) < 0:
                    continue
                cl = int(cl)
                cluster_points.setdefault(cl, []).append(
                    (r['UMAP 1 Scaled'], r['UMAP 2 Scaled'])
                )

    def _size_to_rgb(n, min_sz, max_sz):
        t = (n - min_sz) / (max_sz - min_sz + 1e-9)
        stops = [
            (0.00, (60,  100, 220)),
            (0.25, (60,  200, 210)),
            (0.50, (60,  200, 100)),
            (0.75, (220, 200,  50)),
            (1.00, (220,  60,  60)),
        ]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                return tuple(int(c0[j] + f * (c1[j] - c0[j])) for j in range(3))
        return stops[-1][1]

    centroid_x, centroid_y, centroid_cl, centroid_colors, centroid_labels = [], [], [], [], []

    if cluster_points:
        # Group clusters by their enriched-species color so each group can be
        # drawn as a single merged trace (avoids hundreds of individual traces).
        # Unenriched clusters share one neutral-gray group.
        _UNENRICHED_COLOR = '#666677'
        groups: dict[str, dict] = {}  # hex_color -> {fill_x, fill_y, border_x, border_y}

        for cl, points in sorted(cluster_points.items()):
            if len(points) < 3:
                continue
            try:
                pts = np.array(points)
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
                poly_x = verts[:, 0].tolist() + [verts[0, 0], None]
                poly_y = verts[:, 1].tolist() + [verts[0, 1], None]

                enr = _CLUSTER_ENRICHMENT.get(cl, {})
                dominant_sp = enr.get('Dominant Species', '')
                # Only highlight enrichment if the dominant species is currently visible
                if dominant_sp and dominant_sp in active_species:
                    hex_color = enr.get('Dominant Color', _UNENRICHED_COLOR) or _UNENRICHED_COLOR
                else:
                    hex_color = _UNENRICHED_COLOR

                if hex_color not in groups:
                    groups[hex_color] = {'fill_x': [], 'fill_y': [], 'bord_x': [], 'bord_y': []}
                groups[hex_color]['fill_x'].extend(poly_x)
                groups[hex_color]['fill_y'].extend(poly_y)
                groups[hex_color]['bord_x'].extend(poly_x)
                groups[hex_color]['bord_y'].extend(poly_y)

                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                centroid_x.append(cx)
                centroid_y.append(cy)
                centroid_cl.append(cl)
                centroid_colors.append(hex_color if dominant_sp in active_species else _UNENRICHED_COLOR)
                cluster_name = enr.get('Cluster Name') or f'Cluster {cl}'
                centroid_labels.append(cluster_name)
            except Exception:
                pass

        for hex_color, g in groups.items():
            r_hex = hex_color.lstrip('#')
            r, gr, b = int(r_hex[0:2], 16), int(r_hex[2:4], 16), int(r_hex[4:6], 16)
            is_enriched = hex_color != _UNENRICHED_COLOR
            fill_alpha  = 0.18 if is_enriched else 0.07
            bord_alpha  = 0.60 if is_enriched else 0.25
            fig.add_trace(go.Scatter(
                x=g['fill_x'], y=g['fill_y'],
                fill='toself',
                fillcolor=f'rgba({r},{gr},{b},{fill_alpha})',
                line=dict(color=f'rgba({r},{gr},{b},{bord_alpha})', width=1.5 if is_enriched else 0.8),
                mode='lines',
                hoverinfo='skip',
                showlegend=False,
            ))

    for species_data in active_data:
        df_records = species_data['df']

        # Apply protein filter
        if protein_filter and protein_filter != 'all':
            df_records = [r for r in df_records if r.get('filter_tag') == protein_filter]

        if not df_records:
            # Empty trace to preserve curveNumber indexing
            fig.add_trace(go.Scattergl(
                x=[], y=[], mode='markers',
                name=species_data['name'],
                marker=dict(size=point_size, color=species_data['color'],
                           opacity=opacity, line=dict(width=0)),
                customdata=[]
            ))
            continue

        hover_text = []
        for record in df_records:
            text = f"<b>{species_data['name']}</b><br>"
            text += f"Entry: {record['Entry']}<br>"
            if 'Protein names' in record:
                text += f"Protein: {str(record['Protein names'])[:60]}...<br>"
            cl = record.get('Cluster Label')
            cl_name = record.get('Cluster Name') or (
                _CLUSTER_ENRICHMENT.get(int(cl), {}).get('Cluster Name', '') if cl is not None else '')
            if cl is not None:
                enr = _CLUSTER_ENRICHMENT.get(int(cl) if cl != -1 else -1, {})
                dominant = enr.get('Dominant Species', '')
                text += f"Cluster: {cl_name or cl}"
                if dominant:
                    text += f" <i>(enriched: {dominant})</i>"
            hover_text.append(text)
        fig.add_trace(go.Scattergl(
            x=[r['UMAP 1 Scaled'] for r in df_records],
            y=[r['UMAP 2 Scaled'] for r in df_records],
            mode='markers',
            name=species_data['name'],
            marker=dict(size=point_size, color=species_data['color'],
                       opacity=opacity, line=dict(width=0)),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            customdata=[[r['Entry'], r.get('Cluster Label', -1)] for r in df_records]
        ))
    # Centroid labels on top of everything — large hit target so clicks land reliably.
    # Labels show cluster name; enriched clusters use their species color.
    if centroid_x:
        centroid_hover = []
        for cl, label in zip(centroid_cl, centroid_labels):
            enr = _CLUSTER_ENRICHMENT.get(cl, {})
            dominant = enr.get('Dominant Species', '')
            all_enr_raw = enr.get('All Enriched', '') or ''
            # Only show enrichment hint for species that are currently visible
            visible_enriched = [s for s in all_enr_raw.split(', ') if s and s in active_species]
            if visible_enriched:
                tip = f'<b>{label}</b><br>Enriched: {", ".join(visible_enriched)}<extra></extra>'
            else:
                tip = f'<b>{label}</b> — click to select<extra></extra>'
            centroid_hover.append(tip)

        fig.add_trace(go.Scatter(
            x=centroid_x, y=centroid_y,
            mode='markers+text',
            marker=dict(
                size=28,
                color=[f'rgba(255,255,255,{0.18 if c != "#666677" else 0.04})'
                       for c in centroid_colors],
                line=dict(color=centroid_colors, width=[2 if c != '#666677' else 0.5
                                                        for c in centroid_colors]),
            ),
            text=centroid_labels,
            textfont=dict(size=8, color=centroid_colors),
            textposition='middle center',
            customdata=[[c] for c in centroid_cl],
            hovertemplate=centroid_hover,
            showlegend=False,
        ))

    # Highlight overlay — gold ring on top, transparent fill preserves species colors
    if highlight_data and highlight_data.get('xs'):
        fig.add_trace(go.Scattergl(
            x=highlight_data['xs'],
            y=highlight_data['ys'],
            mode='markers',
            name=highlight_data.get('label', 'Highlighted'),
            marker=dict(
                symbol='circle-open',
                color='#FFD700',
                size=point_size + 7,
                line=dict(color='#FFD700', width=2.5)
            ),
            hoverinfo='skip',
            showlegend=True,
        ))

    margin = 50
    fig.update_layout(
        transition=dict(duration=500, easing='cubic-in-out'),
        title=None,
        xaxis=dict(range=[-150, 150], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-150, 150], showgrid=False, zeroline=False, visible=False,
                  scaleanchor="x", scaleratio=1),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        hovermode='closest',
        dragmode='lasso',
        clickmode='event+select',
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor='rgba(10, 25, 47, 0.2)',
            borderwidth=1,
            font=dict(size=12, family='"Inter", sans-serif', color='#0a192f')
        ),
        margin=dict(l=20, r=20, t=30, b=50),
        autosize=True,
        hoverlabel=dict(
            bgcolor='rgba(10, 25, 47, 0.95)',
            bordercolor='rgba(100, 255, 218, 0.3)',
            font=dict(family='"Inter", sans-serif', size=12, color='#f1faee')
        )
    )
    fig.update_traces(
        selectedpoints=[],
        selected=dict(marker=dict(opacity=1.0, size=point_size)),
        unselected=dict(marker=dict(opacity=opacity, size=point_size))
    )
    return fig

# Update filter count display
@app.callback(
    Output('filter-count-display', 'children'),
    [Input('protein-filter', 'value'),
     Input('loaded-data', 'data'),
     Input('active-species-store', 'data')]
)
def update_filter_count(protein_filter, loaded_data, active_species):
    if not loaded_data:
        return ""
    active_species = active_species or []
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    if not active_data:
        return ""
    _compute_filter_tags(active_data)
    total = 0
    shown = 0
    for species_data in active_data:
        records = species_data['df']
        total += len(records)
        if protein_filter and protein_filter != 'all':
            shown += sum(1 for r in records if r.get('filter_tag') == protein_filter)
        else:
            shown += len(records)
    if protein_filter and protein_filter != 'all':
        return f"Showing {shown:,}/{total:,} proteins"
    return ""

# Grey out filter radio buttons when only 1 species is active
@app.callback(
    [Output('protein-filter-container', 'style'),
     Output('protein-filter', 'value', allow_duplicate=True)],
    [Input('active-species-store', 'data')],
    [State('protein-filter', 'value')],
    prevent_initial_call=True
)
def toggle_filter_enabled(active_species, current_filter):
    if not active_species or len(active_species) <= 1:
        return {'opacity': 0.35, 'pointerEvents': 'none'}, 'all'
    return {}, current_filter

def _render_table(rows, header_text, header_color='#64ffda'):
    """Render the protein selection DataTable with a header."""
    return html.Div([
        html.Div(header_text,
                style={'fontSize': 13, 'fontWeight': '600', 'marginBottom': 15,
                      'color': header_color, 'fontFamily': '"JetBrains Mono", monospace',
                      'letterSpacing': '1px'}),
        dash_table.DataTable(
            id='protein-datatable',
            data=rows,
            columns=[
                {'name': 'Entry', 'id': 'Entry'},
                {'name': 'Organism', 'id': 'Organism'},
                {'name': 'Protein Name', 'id': 'Protein Name'},
                {'name': 'Sequence', 'id': 'Sequence'},
                {'name': 'Cluster', 'id': 'Cluster'},
                {'name': 'Domains', 'id': 'Domains'},
                {'name': 'HMM Descriptions', 'id': 'HMM Descriptions'},
                {'name': 'Pfam IDs', 'id': 'Pfam IDs'},
                {'name': 'Domain Ranges', 'id': 'Domain Ranges'},
            ],
            style_table={'overflowX': 'auto', 'borderRadius': '8px'},
            style_cell={
                'textAlign': 'left', 'padding': '12px 10px',
                'fontFamily': '"JetBrains Mono", monospace', 'fontSize': 11,
                'overflow': 'hidden', 'textOverflow': 'ellipsis',
                'maxWidth': '180px', 'backgroundColor': 'transparent', 'border': 'none'
            },
            style_header={
                'backgroundColor': 'rgba(100, 255, 218, 0.1)', 'fontWeight': '600',
                'color': '#64ffda', 'borderBottom': '1px solid rgba(100, 255, 218, 0.2)',
                'fontSize': 10, 'textTransform': 'uppercase', 'letterSpacing': '1px'
            },
            style_data={
                'backgroundColor': 'rgba(10, 25, 47, 0.4)', 'color': '#f1faee',
                'borderBottom': '1px solid rgba(100, 255, 218, 0.1)', 'cursor': 'pointer'
            },
            css=[
                {'selector': 'td.dash-cell', 'rule': 'color: #f1faee !important;'},
                {'selector': 'td.dash-cell.focused', 'rule': 'background-color: rgba(100, 255, 218, 0.15) !important; color: #f1faee !important; outline: none !important;'},
                {'selector': 'input.dash-cell-value', 'rule': 'background-color: transparent !important; color: #f1faee !important; caret-color: transparent !important; border: none !important;'},
                {'selector': 'tr:hover td', 'rule': 'color: #f1faee !important;'},
                {'selector': '.dash-spreadsheet-inner tr:hover td::after, .dash-spreadsheet-inner td::after', 'rule': 'display: none !important;'},
                {'selector': '.dash-spreadsheet-inner tr:hover', 'rule': 'background: none !important;'},
            ],
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(10, 25, 47, 0.6)'},
                {'if': {'state': 'active'}, 'backgroundColor': 'rgba(100, 255, 218, 0.15)',
                 'color': '#f1faee', 'border': '1px solid rgba(100, 255, 218, 0.3)',
                 'overflow': 'visible', 'whiteSpace': 'normal', 'height': 'auto'}
            ],
            editable=False, page_size=10, page_action='native',
            sort_action='native', sort_by=[{'column_id': 'Protein Name', 'direction': 'asc'}]
        )
    ])


def _build_protein_rows(entry_ids, loaded_data, active_species):
    """Build table row dicts from a list of (species_name, entry_id) tuples."""
    active_species = active_species or []
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    # Primary lookup from display sample; secondary from full search data for rare proteins
    species_lookup = {}
    for sd in active_data:
        species_lookup[sd['name']] = {rec['Entry']: rec for rec in sd['df']}
    full_lookup = {}
    for sd in active_data:
        full_lookup[sd['name']] = {rec['Entry']: rec for rec in _FULL_SEARCH_DATA.get(sd['name'], [])}

    rows = []
    for species_name, entry_id in entry_ids:
        protein = species_lookup.get(species_name, {}).get(entry_id) \
               or full_lookup.get(species_name, {}).get(entry_id)
        if not protein:
            continue
        ann = ANNOTATIONS.get(entry_id)
        desc_raw = (ann.get('desc', '') if ann else '') or ''
        protein_name = desc_raw.split(' OS=')[0].strip() if desc_raw else '—'
        seq_raw = (ann.get('sequence', '') if ann else '') or ''
        seq_preview = (seq_raw[:20] + '...') if len(seq_raw) > 20 else (seq_raw or '—')
        display = SPECIES_DATA.get(species_name, {}).get('display_name', species_name)
        rows.append({
            'Entry': entry_id,
            'Organism': display,
            'Protein Name': protein_name[:80],
            'Sequence': seq_preview,
            'Cluster': str(protein.get('Cluster Label', 'N/A')),
            'Domains': ann.get('hmm_labels', '') or '—' if ann else '—',
            'HMM Descriptions': ann.get('hmm_descriptions', '') or '—' if ann else '—',
            'Pfam IDs': ann.get('hmm_pfam_ids', '') or '—' if ann else '—',
            'Domain Ranges': ann.get('hmm_ranges', '') or '—' if ann else '—',
        })
    return rows


@app.callback(
    Output('selection-table-container', 'children'),
    [Input('interactive-graph', 'selectedData'),
     Input('highlight-store', 'data')],
    [State('loaded-data', 'data'),
     State('active-species-store', 'data')]
)
def show_selected_proteins(selectedData, highlight_data, loaded_data, active_species):
    if not loaded_data:
        return "No proteins selected. Click or drag to select."

    # Highlight search takes precedence when nothing is lassoed
    if (not selectedData or not selectedData.get('points')) and highlight_data and highlight_data.get('entry_ids'):
        rows = _build_protein_rows(highlight_data['entry_ids'], loaded_data, active_species)
        label = highlight_data.get('label', 'Highlighted')
        header_text = f"{len(rows)} protein{'s' if len(rows) != 1 else ''} — {label}"
        header_color = '#FFD700'
        return _render_table(rows, header_text, header_color)

    if not selectedData or not selectedData.get('points'):
        return "No proteins selected. Click or drag to select."

    try:
        active_species = active_species or []
        active_data = [sd for sd in loaded_data if sd['name'] in active_species]

        # Flat lookup: entry_id -> (species_name, record). Avoids curveNumber entirely —
        # hull/centroid traces are added before species traces so curveNumber is unreliable.
        entry_lookup = {}
        for sd in active_data:
            for record in sd['df']:
                entry_lookup[str(record['Entry'])] = (sd['name'], record)

        selected_proteins = []
        seen = set()
        for point in selectedData['points']:
            customdata = point.get('customdata')
            # Species scatter customdata = [entry_id, cluster_label] (len >= 2, entry is a string)
            # Hull/centroid customdata = [cluster_int] (len == 1) — skip these
            if not customdata or len(customdata) < 2:
                continue
            entry_id = str(customdata[0])
            if entry_id in seen or entry_id not in entry_lookup:
                continue
            seen.add(entry_id)
            species_name, protein = entry_lookup[entry_id]

            if not protein:
                continue

            entry_id_str = str(protein.get('Entry', ''))
            ann = ANNOTATIONS.get(entry_id_str)

            # Use desc from annotations for protein name
            desc_raw = (ann.get('desc', '') if ann else '') or ''
            # Strip OS=... suffix from UniProt desc if present
            protein_name = desc_raw.split(' OS=')[0].strip() if desc_raw else '\u2014'

            # Truncated sequence preview
            seq_raw = (ann.get('sequence', '') if ann else '') or ''
            seq_preview = (seq_raw[:20] + '...') if len(seq_raw) > 20 else (seq_raw or '\u2014')

            selected_proteins.append({
                'Entry': protein.get('Entry', 'N/A'),
                'Organism': SPECIES_DATA.get(species_name, {}).get('display_name', species_name),
                'Protein Name': protein_name[:80],
                'Sequence': seq_preview,
                'Cluster': str(protein.get('Cluster Label', 'N/A')),
                'Domains': ann.get('hmm_labels', '') or '\u2014' if ann else '\u2014',
                'HMM Descriptions': ann.get('hmm_descriptions', '') or '\u2014' if ann else '\u2014',
                'Pfam IDs': ann.get('hmm_pfam_ids', '') or '\u2014' if ann else '\u2014',
                'Domain Ranges': ann.get('hmm_ranges', '') or '\u2014' if ann else '\u2014',
            })
        return _render_table(
            selected_proteins,
            f"{len(selected_proteins)} protein{'s' if len(selected_proteins) != 1 else ''} selected"
        )
    except Exception as e:
        return html.Div(f"Error loading selection: {str(e)}",
                       style={'padding': 15, 'color': '#e63946',
                             'fontFamily': '"JetBrains Mono", monospace',
                             'backgroundColor': 'rgba(230, 57, 70, 0.1)',
                             'borderRadius': 8,
                             'border': '1px solid rgba(230, 57, 70, 0.3)'})



@app.callback(
    [Output('selection-screen', 'style', allow_duplicate=True),
     Output('explorer-screen', 'style', allow_duplicate=True),
     Output('animation-timer', 'n_intervals'),
     Output('animation-timer', 'disabled', allow_duplicate=True),
     Output('gif-load-poller', 'disabled', allow_duplicate=True),
     Output('loading-indicator', 'style', allow_duplicate=True),
     Output('animation-container', 'style', allow_duplicate=True)],
    [Input('back-button', 'n_clicks')],
    prevent_initial_call=True
)
def go_back(n_clicks):
    if n_clicks:
        return ({'display': 'block', 'backgroundColor': 'transparent',
                'minHeight': '100vh', 'paddingTop': 60, 'opacity': 1},
                {'display': 'none', 'opacity': 0},
                0, True, True,
                {'textAlign': 'center', 'padding': '120px 20px'},
                {'display': 'none'})
    raise dash.exceptions.PreventUpdate

# ============= CLUSTER SELECTION CALLBACK =============

@app.callback(
    [Output('selected-cluster-store', 'data'),
     Output('highlight-store', 'data', allow_duplicate=True),
     Output('selected-cluster-badge', 'children'),
     Output('chat-messages-display', 'children', allow_duplicate=True),
     Output('chat-history', 'data', allow_duplicate=True)],
    Input('interactive-graph', 'clickData'),
    [State('loaded-data', 'data'),
     State('active-species-store', 'data'),
     State('chat-messages-display', 'children'),
     State('chat-history', 'data')],
    prevent_initial_call=True,
)
def select_cluster_on_click(click_data, loaded_data, active_species, current_msgs, chat_history):
    if not click_data or not loaded_data or not active_species:
        raise dash.exceptions.PreventUpdate

    # Extract cluster id from click.
    # Hull traces:    customdata = [cl]          (an int)
    # Scatter traces: customdata = [entry_id, cluster_label]  (string, int/-1)
    points = click_data.get('points', [])
    cluster_id = None
    for pt in points:
        cd = pt.get('customdata')
        if cd is None:
            continue
        # Case 1: hull trace — single int element
        if len(cd) == 1:
            try:
                val = int(cd[0])
                if val >= 0:
                    cluster_id = val
                    break
            except (TypeError, ValueError):
                pass
        # Case 2: scatter trace — [entry_id, cluster_label]
        if len(cd) >= 2:
            try:
                val = int(cd[1])
                if val >= 0:
                    cluster_id = val
                    break
            except (TypeError, ValueError):
                pass

    if cluster_id is None:
        raise dash.exceptions.PreventUpdate

    chat_history = chat_history or []
    current_msgs = current_msgs or []

    # Highlight cluster proteins on plot
    highlight_data = _resolve_highlight_query(str(cluster_id), loaded_data, active_species)

    # Build enrichment context for this cluster
    active_data = [sd for sd in loaded_data if sd['name'] in active_species]
    info = compute_cluster_enrichment(active_data, cluster_id=cluster_id)

    # GPT interpretation grounded in RAG literature
    interpretation = _interpret_cluster(info, active_species)

    badge_text = f"Cluster {cluster_id} selected"
    store_data = {'cluster_id': cluster_id, 'info': info, 'interpretation': interpretation}

    msg_div = html.Div(
        dcc.Markdown(interpretation, className='chat-md'),
        className='chat-msg chat-msg-assistant'
    )
    chat_history.append({"role": "assistant", "content": interpretation})
    new_msgs = list(current_msgs) + [msg_div]

    return store_data, highlight_data, badge_text, new_msgs, chat_history


# Fire a window resize event shortly after the graph figure updates so Plotly
# recomputes its container dimensions and fixes the initial aspect-ratio skew.
app.clientside_callback(
    """
    function(figure) {
        if (!figure) return window.dash_clientside.no_update;
        var gd = document.getElementById('interactive-graph');
        if (!gd) return window.dash_clientside.no_update;

        function doResize() {
            if (window.Plotly && gd._fullLayout) {
                Plotly.relayout(gd, {autosize: true});
            }
        }

        // Fire at multiple delays to catch whenever the container settles
        setTimeout(doResize, 100);
        setTimeout(doResize, 400);
        setTimeout(doResize, 900);

        // Also watch for container size changes (catches CSS transition end)
        if (window.ResizeObserver) {
            if (window._plotResizeObserver) { window._plotResizeObserver.disconnect(); }
            var fired = 0;
            window._plotResizeObserver = new ResizeObserver(function() {
                doResize();
                fired++;
                if (fired >= 3) { window._plotResizeObserver.disconnect(); }
            });
            window._plotResizeObserver.observe(gd.parentElement || gd);
        }

        return window.dash_clientside.no_update;
    }
    """,
    Output('interactive-graph', 'id'),
    Input('interactive-graph', 'figure'),
    prevent_initial_call=False,
)

# ============= CHAT CALLBACKS =============

@app.callback(
    Output('chat-panel', 'style'),
    Input('chat-toggle-btn', 'n_clicks'),
    State('chat-panel', 'style'),
    prevent_initial_call=True,
)
def toggle_chat_panel(n_clicks, current_style):
    current_style = current_style or {}
    if current_style.get('display') == 'none':
        current_style['display'] = 'flex'
    else:
        current_style['display'] = 'none'
    return current_style


# Instant feedback: show user message + thinking indicator, clear input
app.clientside_callback(
    """
    function(n_clicks, n_submit, inputValue, currentChildren) {
        if (!inputValue || !inputValue.trim()) {
            return [window.dash_clientside.no_update,
                    window.dash_clientside.no_update,
                    window.dash_clientside.no_update];
        }
        var msg = inputValue.trim();
        var newChildren = [];
        if (currentChildren) {
            for (var i = 0; i < currentChildren.length; i++) {
                newChildren.push(currentChildren[i]);
            }
        }
        newChildren.push({
            type: 'Div', namespace: 'dash_html_components',
            props: {children: msg, className: 'chat-msg chat-msg-user'}
        });
        newChildren.push({
            type: 'Div', namespace: 'dash_html_components',
            props: {children: 'Thinking...', className: 'chat-msg chat-msg-assistant',
                    style: {opacity: 0.5, fontStyle: 'italic'}}
        });
        setTimeout(function() {
            var el = document.getElementById('chat-messages-display');
            if (el) { el.scrollTop = el.scrollHeight; }
        }, 50);
        return [newChildren, '', msg];
    }
    """,
    [Output('chat-messages-display', 'children', allow_duplicate=True),
     Output('chat-input', 'value', allow_duplicate=True),
     Output('pending-query', 'data')],
    [Input('chat-send-btn', 'n_clicks'),
     Input('chat-input', 'n_submit')],
    [State('chat-input', 'value'),
     State('chat-messages-display', 'children')],
    prevent_initial_call=True,
)


@app.callback(
    [Output('chat-messages-display', 'children'),
     Output('chat-history', 'data'),
     Output('highlight-store', 'data', allow_duplicate=True),
     Output('active-species-store', 'data', allow_duplicate=True),
     Output('protein-filter', 'value', allow_duplicate=True),
     Output('point-size', 'value', allow_duplicate=True),
     Output('opacity', 'value', allow_duplicate=True)],
    Input('pending-query', 'data'),
    [State('chat-history', 'data'),
     State('active-species-store', 'data'),
     State('interactive-graph', 'selectedData'),
     State('loaded-data', 'data'),
     State('highlight-store', 'data'),
     State('point-size', 'value'),
     State('opacity', 'value'),
     State('selected-cluster-store', 'data'),
     State('protein-filter', 'value')],
    prevent_initial_call=True,
)
def process_chat_query(pending_query, chat_history, active_species, selected_data, loaded_data, highlight_store, current_size, current_opacity, selected_cluster, protein_filter):
    if not pending_query:
        raise dash.exceptions.PreventUpdate

    chat_history = chat_history or []
    visual_context = build_visual_context(active_species, selected_data, loaded_data)

    # Inject active filter context
    if protein_filter and protein_filter != 'all':
        if protein_filter == 'unique':
            active_data = [sd for sd in (loaded_data or []) if sd['name'] in (active_species or [])]
            _compute_filter_tags(active_data)
            sample = []
            for sd in active_data:
                for rec in sd['df']:
                    if rec.get('filter_tag') == 'unique' and len(sample) < 10:
                        entry = rec.get('Entry', '')
                        ann = ANNOTATIONS.get(str(entry), {})
                        name = (ann.get('desc', '') or '').split(' OS=')[0].strip() or entry
                        sample.append(f"  - {entry} ({sd['name']}): {name[:80]}")
            sample_str = '\n'.join(sample) if sample else '  (none in current view sample)'
            visual_context += (
                f"\n\nActive filter: SPECIES-SPECIFIC proteins only "
                f"(proteins not grouped into any shared cluster — ~33% of each proteome).\n"
                f"Sample species-specific proteins currently shown:\n{sample_str}"
            )
        elif protein_filter == 'shared':
            visual_context += "\n\nActive filter: SHARED proteins only (proteins grouped into clusters present across multiple species)."

    # Inject selected-cluster context so subsequent questions are cluster-aware
    if selected_cluster:
        cl_id = selected_cluster.get('cluster_id')
        info  = selected_cluster.get('info', {})
        interp = selected_cluster.get('interpretation', '')
        if cl_id is not None and info:
            enriched = info.get('enriched_domains', [])
            descs    = info.get('domain_descriptions', {})
            dom_str  = ", ".join(
                f"{d} ({descs.get(d, '')}; {fold:.1f}×)" if descs.get(d) else f"{d} ({fold:.1f}×)"
                for d, fold, _ in enriched
            ) if enriched else "none detected"
            names_str = "; ".join(info.get('sample_names', [])[:10])
            cluster_ctx = (
                f"Selected cluster: Cluster {cl_id} ({info.get('size', '?')} proteins, "
                f"{info.get('sharing', '')})\n"
                f"Enriched domains: {dom_str}\n"
                f"Example proteins: {names_str}"
            )
            if interp:
                cluster_ctx += f"\n\nPrevious interpretation:\n{interp}"
            visual_context = (visual_context + "\n\n" if visual_context else "") + cluster_ctx

    if highlight_store and highlight_store.get('entry_ids'):
        label = highlight_store.get('label', 'Highlighted')
        rows = _build_protein_rows(highlight_store['entry_ids'], loaded_data, active_species)
        if rows:
            lines = [f"Currently highlighted on plot: {label} ({len(rows)} proteins)"]
            for r in rows[:20]:
                lines.append(
                    f"  - {r['Entry']} ({r['Organism']}): {r['Protein Name']}, "
                    f"Cluster: {r['Cluster']}, Domains: {r.get('Domains', '—')}"
                )
            if len(rows) > 20:
                lines.append(f"  ... and {len(rows) - 20} more")
            visual_context = (visual_context + "\n\n" if visual_context else "") + "\n".join(lines)

    if loaded_data:
        dataset_ctx = _dataset_lookup_context(pending_query, loaded_data, active_species)
        if dataset_ctx:
            visual_context = (visual_context + "\n\n" if visual_context else "") + dataset_ctx

    new_highlight = dash.no_update
    new_active_species = dash.no_update
    new_filter = dash.no_update
    new_size = dash.no_update
    new_opacity = dash.no_update
    mode, text = route_message(chat_history, pending_query, visual_context)

    def _render(history):
        return [
            html.Div(m['content'], className='chat-msg chat-msg-user') if m['role'] == 'user'
            else html.Div(dcc.Markdown(m['content'], className='chat-md'), className='chat-msg chat-msg-assistant')
            for m in history
        ]

    if mode == "control":
        new_filter, new_size, new_opacity, answer = _resolve_ui_command(text, current_size, current_opacity)
        chat_history.append({"role": "user", "content": pending_query})
        chat_history.append({"role": "assistant", "content": answer})
        return _render(chat_history), chat_history, new_highlight, new_active_species, new_filter, new_size, new_opacity

    if mode == "species":
        new_list, answer = _resolve_species_command(text, active_species)
        if new_list is not None:
            new_active_species = new_list
        chat_history.append({"role": "user", "content": pending_query})
        chat_history.append({"role": "assistant", "content": answer})
        return _render(chat_history), chat_history, new_highlight, new_active_species, new_filter, new_size, new_opacity

    if mode == "cluster_search":
        answer, search_highlight = _cluster_search(text, loaded_data, active_species)
        if search_highlight:
            new_highlight = search_highlight
        chat_history.append({"role": "user", "content": pending_query})
        chat_history.append({"role": "assistant", "content": answer})
        return _render(chat_history), chat_history, new_highlight, new_active_species, new_filter, new_size, new_opacity

    # If the query is about a single specific cluster (any mode except cluster_search),
    # run the full interpretation — same analysis as clicking a hull.
    if mode not in ("control", "species") and loaded_data and active_species:
        query_clusters = _extract_cluster_ids_from_text(pending_query)
        if len(query_clusters) == 1:
            cl_id = query_clusters[0]
            active_data_local = [sd for sd in loaded_data if sd['name'] in active_species]
            info = compute_cluster_enrichment(active_data_local, cluster_id=cl_id)
            if info and info.get('size', 0) > 0:
                interpretation = _interpret_cluster(info, active_species)
                new_highlight = _resolve_highlight_query(str(cl_id), loaded_data, active_species)
                chat_history.append({"role": "user", "content": pending_query})
                chat_history.append({"role": "assistant", "content": interpretation})
                return _render(chat_history), chat_history, new_highlight, new_active_species, new_filter, new_size, new_opacity

    if mode == "highlight":
        highlight_data = _resolve_highlight_query(text, loaded_data, active_species)
        new_highlight = highlight_data
        if highlight_data:
            count = len(highlight_data['xs'])
            answer = (
                f"I've highlighted **{highlight_data['label']}** on the plot — "
                f"{count} protein{'s' if count != 1 else ''} marked with gold rings. 🎯"
            )
        else:
            answer = (
                f"I couldn't find **{text}** in the currently loaded data. "
                "Make sure the species containing that cluster or protein is active, "
                "or try the search bar."
            )
    elif mode == "rag":
        raw, sources_block = query_rag(text, visual_context)
        if raw.startswith("Error:"):
            answer = answer_with_context(text, visual_context, chat_history) if visual_context else chat_reply(pending_query, chat_history, visual_context)
        else:
            answer = rewrite_as_guide(raw, pending_query, chat_history, visual_context)
            if sources_block:
                answer = answer + "\n\n" + sources_block
    elif mode == "visual":
        answer = answer_with_context(text, visual_context, chat_history)
    else:
        answer = chat_reply(pending_query, chat_history, visual_context)

    # Auto-highlight clusters if nothing is already highlighted.
    # Priority: 1) clusters mentioned in the user's own query (most reliable)
    #           2) clusters mentioned in the AI answer
    #           3) protein name extracted from a RAG query
    #           4) currently selected cluster (re-highlight after any follow-up)
    if new_highlight is dash.no_update and loaded_data:
        mentioned = (
            _extract_cluster_ids_from_text(pending_query)
            or _extract_cluster_ids_from_text(answer)
        )
        if mentioned:
            auto_hl = _merge_highlight_for_clusters(mentioned, loaded_data, active_species)
            if auto_hl:
                new_highlight = auto_hl
        elif mode == "rag" and loaded_data:
            # Auto-highlight the protein the user is asking about
            protein_term = _extract_protein_search_term(pending_query)
            if protein_term:
                auto_hl = _resolve_highlight_query(protein_term, loaded_data, active_species)
                if auto_hl and auto_hl.get('entry_ids') and len(auto_hl['entry_ids']) <= 200:
                    new_highlight = auto_hl
        elif selected_cluster and selected_cluster.get('cluster_id') is not None:
            # Re-affirm the selected cluster highlight so it stays visible after Q&A
            cl_id = selected_cluster['cluster_id']
            auto_hl = _resolve_highlight_query(str(cl_id), loaded_data, active_species)
            if auto_hl:
                new_highlight = auto_hl

    chat_history.append({"role": "user", "content": pending_query})
    chat_history.append({"role": "assistant", "content": answer})
    return _render(chat_history), chat_history, new_highlight, new_active_species, new_filter, new_size, new_opacity


# ── Phylogenetic tree panel ────────────────────────────────────────────────────

def _build_selection_phylo_figure(active_species):
    """
    Full-page phylo tree for the species-selection screen.
    Tool species show their animal image at the leaf; non-tool species get a dim dot.
    Hover customdata = [sp_key, '/animal_pics/<file>'] for clientside preview.
    """
    BG     = '#08111f'
    BRANCH = 'rgba(100,255,218,0.30)'

    TOOL_KEYS = {k: SPECIES_DATA[k]['color'] for k in SPECIES_DATA}
    active_set = set(active_species or [])

    LEAVES = [
        ('human',      9.0, 'Human',               False, None),
        ('hippo',      8.0, 'Hippopotamus',         True,  'Hippopotamus'),
        ('graywhale',  7.0, 'Gray Whale',           True,  'Gray Whale'),
        ('spermwhale', 6.0, 'Sperm Whale',          False, None),
        ('orca',       5.0, 'Orca',                 True,  'Orca'),
        ('bottlenose', 4.0, 'Bottlenose Dolphin',   True,  'Bottlenose Dolphin'),
        ('polarbear',  3.0, 'Polar Bear',           True,  'Polar Bear'),
        ('walrus',     2.0, 'Walrus',               False, None),
        ('harborseal', 1.0, 'Harbor Seal',          True,  'Harbor Seal'),
        ('sealion',    0.0, 'Sea Lion',             True,  'Sea Lion'),
    ]

    # Same tree geometry as _build_phylo_figure
    LEAF_X = 5.8
    X_ROOT, X_LAUR = 0.60, 1.60
    X_CETARTIO, X_CET, X_ODONT, X_DELPH = 2.75, 3.95, 5.10, 5.55
    X_CARN, X_PINN, X_PHOC_OTA = 2.75, 3.95, 5.10

    y_delph    = 4.5
    y_odont    = (6.0 + y_delph) / 2
    y_cet      = (7.0 + y_odont) / 2
    y_cetartio = (8.0 + y_cet)   / 2
    y_phoc_ota = 0.5
    y_pinn     = (2.0 + y_phoc_ota) / 2
    y_carn     = (3.0 + y_pinn)     / 2
    y_laur     = (y_cetartio + y_carn) / 2
    y_root     = (9.0 + y_laur) / 2

    segs = [
        (X_ROOT, y_laur, X_ROOT, 9.0), (X_ROOT, 9.0, LEAF_X, 9.0),
        (X_ROOT, y_laur, X_LAUR, y_laur),
        (X_LAUR, y_carn, X_LAUR, y_cetartio),
        (X_LAUR, y_cetartio, X_CETARTIO, y_cetartio),
        (X_LAUR, y_carn, X_CARN, y_carn),
        (X_CETARTIO, y_cet, X_CETARTIO, 8.0),
        (X_CETARTIO, 8.0, LEAF_X, 8.0),
        (X_CETARTIO, y_cet, X_CET, y_cet),
        (X_CET, y_odont, X_CET, 7.0), (X_CET, 7.0, LEAF_X, 7.0),
        (X_CET, y_odont, X_ODONT, y_odont),
        (X_ODONT, y_delph, X_ODONT, 6.0), (X_ODONT, 6.0, LEAF_X, 6.0),
        (X_ODONT, y_delph, X_DELPH, y_delph),
        (X_DELPH, 4.0, X_DELPH, 5.0), (X_DELPH, 5.0, LEAF_X, 5.0), (X_DELPH, 4.0, LEAF_X, 4.0),
        (X_CARN, y_pinn, X_CARN, 3.0), (X_CARN, 3.0, LEAF_X, 3.0),
        (X_CARN, y_pinn, X_PINN, y_pinn),
        (X_PINN, y_phoc_ota, X_PINN, 2.0), (X_PINN, 2.0, LEAF_X, 2.0),
        (X_PINN, y_phoc_ota, X_PHOC_OTA, y_phoc_ota),
        (X_PHOC_OTA, 0.0, X_PHOC_OTA, 1.0),
        (X_PHOC_OTA, 1.0, LEAF_X, 1.0), (X_PHOC_OTA, 0.0, LEAF_X, 0.0),
    ]
    shapes = [dict(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                   line=dict(color=BRANCH, width=2)) for x0,y0,x1,y1 in segs]

    # Animal images placed to the right of LEAF_X
    IMG_LEFT   = LEAF_X + 0.15   # left anchor x
    IMG_W      = 2.4              # sizex (data coords)
    IMG_H      = 0.82             # sizey (data coords, < 1.0 leaf spacing)
    IMG_CENTER = IMG_LEFT + IMG_W / 2

    layout_images = []
    annots = []
    node_x, node_y, node_color, node_size, node_custom = [], [], [], [], []

    for key, y, label, is_tool, sp_key in LEAVES:
        if is_tool and sp_key:
            is_active = sp_key in active_set
            img_file  = Path(SPECIES_DATA[sp_key]['image']).name
            img_url   = f'/animal_pics/{img_file}'
            color     = TOOL_KEYS[sp_key]
            opacity   = 1.0 if is_active else 0.38

            layout_images.append(dict(
                source=img_url,
                x=IMG_LEFT, y=y,
                xref='x', yref='y',
                sizex=IMG_W, sizey=IMG_H,
                xanchor='left', yanchor='middle',
                sizing='contain',
                opacity=opacity,
                layer='above',
            ))

            # Name below image
            annots.append(dict(
                x=IMG_CENTER, y=y - IMG_H / 2 - 0.06,
                text=f"<b>{label}</b>" if is_active else label,
                font=dict(
                    size=10 if is_active else 9,
                    color=color if is_active else 'rgba(168,218,220,0.45)',
                    family='JetBrains Mono, monospace',
                ),
                showarrow=False, xanchor='center', yanchor='top',
            ))

            # Invisible large marker for hover/click detection
            node_x.append(IMG_CENTER)
            node_y.append(y)
            node_color.append(color)
            node_size.append(60)
            node_custom.append([sp_key, img_url])
        else:
            # Non-tool: small dim dot + text label
            node_x.append(LEAF_X)
            node_y.append(y)
            node_color.append('rgba(100,255,218,0.15)')
            node_size.append(5)
            node_custom.append([None, None])
            annots.append(dict(
                x=LEAF_X + 0.15, y=y,
                text=label,
                font=dict(size=8, color='rgba(168,218,220,0.25)', family='JetBrains Mono, monospace'),
                showarrow=False, xanchor='left', yanchor='middle',
            ))

    # Clade labels
    for cx, cy, txt in [
        (X_CET + 0.05,      y_cet + 0.12,      '<i>Cetacea</i>'),
        (X_PINN + 0.05,     y_pinn + 0.12,     '<i>Pinnipedia</i>'),
        (X_CETARTIO + 0.05, y_cetartio + 0.12, '<i>Cetartiodactyla</i>'),
        (X_CARN + 0.05,     y_carn + 0.12,     '<i>Carnivora</i>'),
    ]:
        annots.append(dict(
            x=cx, y=cy, text=txt,
            font=dict(size=9, color='rgba(100,255,218,0.55)', family='Inter, sans-serif'),
            showarrow=False, xanchor='left',
        ))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(color=node_color, size=node_size, opacity=0),
        customdata=node_custom,
        hovertemplate='%{customdata[0]}<extra></extra>',
        showlegend=False,
    )

    fig = go.Figure(data=[node_trace])
    fig.update_layout(
        shapes=shapes,
        annotations=annots,
        images=layout_images,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[-0.2, IMG_LEFT + IMG_W + 0.3]),
        yaxis=dict(visible=False, range=[-0.9, 9.9]),
        showlegend=False,
        hovermode='closest',
        height=580,
        dragmode=False,
    )
    return fig


def _build_phylo_figure(active_species, height=220):
    """Return a Plotly figure of the 10-taxon mammal tree, with active species highlighted."""
    import plotly.graph_objects as go

    BG        = '#08111f'
    BRANCH    = 'rgba(100,255,218,0.25)'
    LABEL_CLR = '#a8dadc'
    GRAY_NODE = 'rgba(100,255,218,0.18)'

    # Tool species → app color
    TOOL_KEYS = {k: SPECIES_DATA[k]['color'] for k in SPECIES_DATA}
    # Map leaf key → display info
    LEAVES = [
        ('human',      9.0, 'Human',               False, None),
        ('hippo',      8.0, 'Hippopotamus',         True,  'Hippopotamus'),
        ('graywhale',  7.0, 'Gray Whale',           True,  'Gray Whale'),
        ('spermwhale', 6.0, 'Sperm Whale',          False, None),
        ('orca',       5.0, 'Orca',                 True,  'Orca'),
        ('bottlenose', 4.0, 'Bottlenose Dolphin',   True,  'Bottlenose Dolphin'),
        ('polarbear',  3.0, 'Polar Bear',           True,  'Polar Bear'),
        ('walrus',     2.0, 'Walrus',               False, None),
        ('harborseal', 1.0, 'Harbor Seal',          True,  'Harbor Seal'),
        ('sealion',    0.0, 'Sea Lion',             True,  'Sea Lion'),
    ]

    active_set = set(active_species or [])

    # Geometry (same as matplotlib figure)
    LEAF_X = 6.8
    X_ROOT, X_LAUR = 0.60, 1.60
    X_CETARTIO, X_CET, X_ODONT, X_DELPH = 2.75, 3.95, 5.10, 6.05
    X_CARN, X_PINN, X_PHOC_OTA = 2.75, 3.95, 5.10

    y_delph    = 4.5
    y_odont    = (6.0 + y_delph) / 2
    y_cet      = (7.0 + y_odont) / 2
    y_cetartio = (8.0 + y_cet)   / 2
    y_phoc_ota = 0.5
    y_pinn     = (2.0 + y_phoc_ota) / 2
    y_carn     = (3.0 + y_pinn)     / 2
    y_laur     = (y_cetartio + y_carn) / 2
    y_root     = (9.0 + y_laur) / 2

    # Segments: list of (x0,y0, x1,y1)
    segs = [
        # outgroup
        (X_ROOT, y_laur, X_ROOT, 9.0), (X_ROOT, 9.0, LEAF_X, 9.0),
        # root → laur
        (X_ROOT, y_laur, X_LAUR, y_laur),
        # laur
        (X_LAUR, y_carn, X_LAUR, y_cetartio),
        (X_LAUR, y_cetartio, X_CETARTIO, y_cetartio),
        (X_LAUR, y_carn, X_CARN, y_carn),
        # cetartio
        (X_CETARTIO, y_cet, X_CETARTIO, 8.0),
        (X_CETARTIO, 8.0, LEAF_X, 8.0),
        (X_CETARTIO, y_cet, X_CET, y_cet),
        # cetacea
        (X_CET, y_odont, X_CET, 7.0), (X_CET, 7.0, LEAF_X, 7.0),
        (X_CET, y_odont, X_ODONT, y_odont),
        # odontoceti
        (X_ODONT, y_delph, X_ODONT, 6.0), (X_ODONT, 6.0, LEAF_X, 6.0),
        (X_ODONT, y_delph, X_DELPH, y_delph),
        # delphinidae
        (X_DELPH, 4.0, X_DELPH, 5.0), (X_DELPH, 5.0, LEAF_X, 5.0), (X_DELPH, 4.0, LEAF_X, 4.0),
        # carnivora
        (X_CARN, y_pinn, X_CARN, 3.0), (X_CARN, 3.0, LEAF_X, 3.0),
        (X_CARN, y_pinn, X_PINN, y_pinn),
        # pinnipedia
        (X_PINN, y_phoc_ota, X_PINN, 2.0), (X_PINN, 2.0, LEAF_X, 2.0),
        (X_PINN, y_phoc_ota, X_PHOC_OTA, y_phoc_ota),
        (X_PHOC_OTA, 0.0, X_PHOC_OTA, 1.0),
        (X_PHOC_OTA, 1.0, LEAF_X, 1.0), (X_PHOC_OTA, 0.0, LEAF_X, 0.0),
    ]

    shapes = []
    for x0, y0, x1, y1 in segs:
        shapes.append(dict(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color=BRANCH, width=1.5)))

    # Leaf nodes + labels
    node_x, node_y, node_colors, node_sizes, node_text, node_custom = [], [], [], [], [], []
    for key, y, label, is_tool, sp_key in LEAVES:
        is_active   = sp_key in active_set if sp_key else False
        is_outgroup = key == 'human'
        if is_tool and sp_key:
            color = TOOL_KEYS.get(sp_key, '#64ffda')
        else:
            color = 'rgba(100,255,218,0.25)'
        opacity = 1.0 if is_active else (0.25 if is_tool else 0.15)
        node_x.append(LEAF_X)
        node_y.append(y)
        node_colors.append(color)
        node_sizes.append(10 if is_active else 6)
        # customdata: species key for toggling (None if not a tool species)
        node_custom.append(sp_key)
        node_text.append(
            f"<b style='color:{color}'>{label}</b>" if is_active
            else f"<span style='color:rgba(168,218,220,0.4)'>{label}</span>"
        )

    # Clade labels
    annots = [
        dict(x=X_CET+0.05, y=y_cet+0.08, text='<i>Cetacea</i>',
             font=dict(size=8, color='rgba(100,255,218,0.5)'), showarrow=False, xanchor='left'),
        dict(x=X_PINN+0.05, y=y_pinn+0.08, text='<i>Pinnipedia</i>',
             font=dict(size=8, color='rgba(100,255,218,0.5)'), showarrow=False, xanchor='left'),
        dict(x=X_CETARTIO+0.05, y=y_cetartio+0.08, text='<i>Cetartiodactyla</i>',
             font=dict(size=8, color='rgba(100,255,218,0.5)'), showarrow=False, xanchor='left'),
        dict(x=X_CARN+0.05, y=y_carn+0.08, text='<i>Carnivora</i>',
             font=dict(size=8, color='rgba(100,255,218,0.5)'), showarrow=False, xanchor='left'),
    ]

    # Label trace (text to the right of nodes)
    label_trace = go.Scatter(
        x=[LEAF_X + 0.25] * len(LEAVES),
        y=[l[1] for l in LEAVES],
        mode='text',
        text=[l[2] for l in LEAVES],
        textfont=dict(size=9, family='JetBrains Mono, monospace'),
        textposition='middle right',
        hoverinfo='skip',
    )
    # Update label colors
    label_trace.textfont = dict(
        size=9,
        family='JetBrains Mono, monospace',
        color=[
            (TOOL_KEYS.get(l[4], '#64ffda') if l[3] and l[4] in active_set
             else ('rgba(168,218,220,0.35)' if not l[3] else 'rgba(168,218,220,0.6)'))
            for l in LEAVES
        ]
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=0),
            opacity=[1.0 if s in active_set else (0.3 if LEAVES[i][3] else 0.15)
                     for i, s in enumerate([l[4] for l in LEAVES])],
        ),
        customdata=node_custom,
        hovertemplate='%{customdata}<extra></extra>',
        text=[l[2] for l in LEAVES],
    )

    fig = go.Figure(data=[label_trace, node_trace])
    fig.update_layout(
        shapes=shapes,
        annotations=annots,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(l=10, r=120, t=6, b=6),
        xaxis=dict(visible=False, range=[-0.2, 11.5]),
        yaxis=dict(visible=False, range=[-0.7, 9.7]),
        showlegend=False,
        hovermode='closest',
        font=dict(family='JetBrains Mono, monospace', color=LABEL_CLR),
        height=height,
    )
    return fig


@app.callback(
    Output('phylo-panel', 'style'),
    Input('phylo-toggle-btn', 'n_clicks'),
    State('phylo-panel', 'style'),
    prevent_initial_call=True,
)
def toggle_phylo_panel(n_clicks, current_style):
    style = dict(current_style)
    style['display'] = 'none' if style.get('display') != 'none' else 'block'
    return style


@app.callback(
    Output('phylo-tree-graph', 'figure'),
    Input('active-species-store', 'data'),
)
def update_phylo_tree(active_species):
    return _build_phylo_figure(active_species)


@app.callback(
    Output('active-species-store', 'data', allow_duplicate=True),
    Input('phylo-tree-graph', 'clickData'),
    State('active-species-store', 'data'),
    prevent_initial_call=True,
)
def phylo_click_toggle(click_data, active_species):
    if not click_data:
        raise dash.exceptions.PreventUpdate
    sp_key = click_data['points'][0].get('customdata')
    if not sp_key or sp_key not in SPECIES_DATA:
        raise dash.exceptions.PreventUpdate
    active = list(active_species or [])
    if sp_key in active:
        if len(active) <= 1:
            raise dash.exceptions.PreventUpdate
        active.remove(sp_key)
    else:
        active.append(sp_key)
    return active


# ── Selection-screen phylo tree ────────────────────────────────────────────────

@app.callback(
    Output('selection-phylo-tree', 'figure'),
    Input('selected-species-store', 'data'),
)
def update_selection_phylo(selected_species):
    return _build_selection_phylo_figure(selected_species or [])


# Clientside: hover on tree → update preview image (zero server round-trip)
app.clientside_callback(
    """
    function(hoverData) {
        if (!hoverData || !hoverData.points || !hoverData.points.length) {
            return ['Hover an animal',
                    '',
                    {maxWidth:'100%', maxHeight:'260px', objectFit:'contain',
                     display:'none', borderRadius:'12px',
                     filter:'drop-shadow(0 0 20px rgba(100,255,218,0.3))'}];
        }
        var cd = hoverData.points[0].customdata;
        if (!cd || !cd[0]) {
            return ['Hover an animal', '',
                    {maxWidth:'100%', maxHeight:'260px', objectFit:'contain',
                     display:'none', borderRadius:'12px',
                     filter:'drop-shadow(0 0 20px rgba(100,255,218,0.3))'}];
        }
        var name = cd[0];
        var url  = cd[1] || '';
        return [name, url,
                {maxWidth:'100%', maxHeight:'260px', objectFit:'contain',
                 display: url ? 'block' : 'none', borderRadius:'12px',
                 filter:'drop-shadow(0 0 20px rgba(100,255,218,0.3))'}];
    }
    """,
    [Output('selection-hover-name', 'children'),
     Output('selection-hover-img',  'src'),
     Output('selection-hover-img',  'style')],
    Input('selection-phylo-tree', 'hoverData'),
)


@app.callback(
    [Output(f'btn-{sp}', 'n_clicks') for sp in SPECIES_DATA.keys()],
    Input('selection-phylo-tree', 'clickData'),
    [State(f'btn-{sp}', 'n_clicks') for sp in SPECIES_DATA.keys()],
    prevent_initial_call=True,
)
def selection_tree_click(click_data, *n_clicks_list):
    if not click_data:
        raise dash.exceptions.PreventUpdate
    sp_key = click_data['points'][0].get('customdata')
    if not sp_key or sp_key not in SPECIES_DATA:
        raise dash.exceptions.PreventUpdate
    keys = list(SPECIES_DATA.keys())
    counts = list(n_clicks_list)
    idx = keys.index(sp_key)
    counts[idx] = (counts[idx] or 0) + 1
    return counts


server = app.server

if __name__ == '__main__':
    import os
    print("\n" + "="*60)
    print("Marine Mammal Proteome Explorer - Ocean Edition")
    print("="*60)
    # Pre-load the vectorstore in the background so the server starts immediately
    import threading
    threading.Thread(target=_load_vectorstore, daemon=True).start()
    port = int(os.environ.get('PORT', 7860))
    host = '0.0.0.0'
    print(f"Open browser: http://{host}:{port}/")
    print("\nFeatures:")
    print("  - Deep ocean theme with bioluminescent accents")
    print("  - Animated wave background with floating particles")
    print("  - Glassmorphism UI with smooth transitions")
    print("  - Interactive UMAP proteome visualization")
    print("="*60 + "\n")
    app.run(debug=False, host=host, port=port)
