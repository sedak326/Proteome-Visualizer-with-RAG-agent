"""
Marine Mammal Proteome Explorer - Ocean Theme Edition
User Flow: Select Animals (with icons) → Watch Animation → Explore Proteomes
Deep sea ocean theme with bioluminescent accents
"""

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
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

    if selected_data and selected_data.get('points'):
        # Must match trace order in update_interactive_graph: loaded_data order filtered to active
        if loaded_data:
            active_ordered = [sd['name'] for sd in loaded_data if sd['name'] in active_species]
        else:
            active_ordered = [s for s in SPECIES_DATA.keys() if s in active_species]

        # Single pass — collect everything needed for aggregation and sampling
        cluster_species = {}   # cluster_label -> set of species keys
        species_counts  = {}   # species key -> int
        domain_counts   = {}   # domain label -> int (across all proteins)
        cluster_counts  = {}   # cluster_label -> int
        sample_points   = []   # list of formatted lines for the representative sample

        for point in selected_data['points']:
            curve_num = point.get('curveNumber', 0)
            if curve_num >= len(active_ordered):
                continue
            species = active_ordered[curve_num]
            species_counts[species] = species_counts.get(species, 0) + 1

            hover = point.get('text', '')
            cluster_label = ''
            if 'Cluster:' in hover:
                cluster_label = hover.split('Cluster:')[-1].strip().split('<')[0].strip()
            if cluster_label:
                cluster_species.setdefault(cluster_label, set()).add(species)
                cluster_counts[cluster_label] = cluster_counts.get(cluster_label, 0) + 1

            customdata = point.get('customdata', [])
            if not customdata:
                continue
            entry_id = str(customdata[0])
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
                    "Output only one line starting with QUERY:, VISUAL:, or CHAT:"
                )},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            max_tokens=200,
        )
        result = resp.choices[0].message.content.strip()
        if result.startswith("QUERY:"):
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
            "You are a nerdy scientist who is genuinely fascinated by proteins and what's on screen. "
            "You are the user's personal lab guide on a proteome visualization portal. "
            "You find this stuff interesting and that comes through naturally — but you don't pepper "
            "responses with 'Wow!', 'Amazing!', or hollow enthusiasm. Let the science speak.\n"
            "The user is viewing a UMAP plot where each dot is a protein, colored by species. "
            "Proteins that cluster together share similar structure or function.\n"
            "You receive a spatial summary showing protein counts per region of the plot "
            "(top-left, center, bottom-right, etc.) and which regions have species overlap. "
            "Use this to reference what the user actually sees — e.g. 'See that dense cluster "
            "in the top-left? That's mostly sea lion proteins!' Only mention regions where "
            "something interesting is happening. NEVER guess locations — only reference "
            "regions described in the spatial data.\n"
            "\n"
            "CRITICAL — stay grounded in the data:\n"
            "- ONLY state things that are directly supported by the data provided to you "
            "(spatial summary, selected proteins, cluster labels, domain annotations).\n"
            "- If the data shows something (e.g. no unique clusters for a species), describe "
            "WHAT you see, but do NOT fabricate explanations for WHY. Instead, say what the "
            "data shows and suggest how to investigate further (e.g. 'The data shows all Gray "
            "Whale clusters are shared with Orca — try selecting one of those shared clusters "
            "to see which proteins they have in common!').\n"
            "- NEVER make up evolutionary explanations, biological mechanisms, or causal "
            "reasoning that isn't in the data. If you don't know why, say so honestly.\n"
            "- It's OK to say 'That's interesting — I'm not sure why, but here's what we "
            "could try to find out...'\n"
            "\n"
            "Style rules:\n"
            "- Use approachable, high-school-level language. If you mention jargon, explain it briefly.\n"
            "- Be concise: 3-4 sentences max.\n"
            "- Show genuine excitement about interesting patterns.\n"
            "- Use markdown formatting: **bold** for key terms, *italic* for emphasis, "
            "and emojis to add personality (e.g. \U0001F9EC, \U0001F52C, \U0001F433, \U0001F3AF). "
            "Use line breaks between thoughts for readability.\n"
            "- After answering, suggest ONE concrete next step: either a UI action "
            "(e.g., \"Try lassoing that cluster!\", \"Toggle another species to compare!\") "
            "OR a related topic you can explain.\n"
            "- Don't list raw protein IDs."
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


def query_rag(query_text):
    """Retrieve relevant chunks from Chroma and generate an answer via GPT-4o-mini."""
    try:
        vs = _load_vectorstore()
        retrieved = vs.similarity_search(query_text, k=RAG_TOP_K)

        if not retrieved:
            context = ""
        else:
            context = "\n\n---\n\n".join(d.page_content for d in retrieved)

        resp = _openai_client.chat.completions.create(
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
                    "'This is from general knowledge, not the specific literature.' "
                    "Never fabricate citations or paper titles."
                )},
                {"role": "user", "content": (
                    f"Context:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
                    if context else
                    f"Question: {query_text}\n\nAnswer:"
                )},
            ],
            temperature=0,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


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
                    "You are a nerdy scientist — the user's personal lab guide "
                    "on a marine-mammal proteome explorer. Rewrite the provided answer so it:\n"
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
            "You are a nerdy scientist — the user's personal lab guide "
            "on a marine-mammal proteome explorer. You're warm, curious, and genuinely interested "
            "in what the user is exploring. You do not overreact — avoid 'Wow!', 'Amazing!', "
            "'Oh cool!' and similar hollow exclamations. Express interest through what you say, not how many exclamation marks you use.\n"
            "You can see exactly what the user sees on their screen (species displayed, "
            "spatial layout, selected proteins). Use this to give context-aware replies.\n"
            "You remember everything discussed so far in this conversation. If the user "
            "refers back to something ('tell me more about that', 'why?', 'what do you mean?', "
            "'okay I did it'), look at the conversation history AND what's on screen "
            "to respond meaningfully.\n"
            "Keep replies concise (2-3 sentences).\n"
            "Use markdown: **bold** for key terms, *italic* for emphasis, "
            "emojis for personality (\U0001F9EC, \U0001F52C, \U0001F433, \U0001F3AF). "
            "Use line breaks between thoughts for readability.\n"
            "IMPORTANT: Only state things supported by the data on screen or the conversation "
            "history. NEVER fabricate evolutionary explanations or biological mechanisms. "
            "If you don't know why something is the way it is, say so honestly and suggest "
            "how to investigate further.\n"
            "If the conversation allows, gently suggest something to explore next — "
            "a topic or a tool in the UI."
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
    'King Cobra': {
        'csv': str(BASE_DIR / 'umap_output/King Cobra_umap.csv'),
        'image': str(BASE_DIR / 'animal_pics/cobra.png'),
        'color': '#BFAA00',
        'glow': 'rgba(191, 170, 0, 0.4)',
        'display_name': 'King Cobra',
        'key': 'cobra'
    }
}

SAMPLE_SIZE = 3000
PRERENDERED_DIR = BASE_DIR / 'prerendered_animations'

# Annotated CSV files per species (for protein classification/domain info)
ANNOTATION_FILES = {
    'bottlenose': [str(BASE_DIR / 'annotated/bottlenose_part1_annotated.csv'), str(BASE_DIR / 'annotated/bottlenose_part2_annotated.csv')],
    'graywhale': [str(BASE_DIR / 'annotated/graywhale_part1_annotated.csv'), str(BASE_DIR / 'annotated/graywhale_part2_annotated.csv')],
    'harborseal': [str(BASE_DIR / 'annotated/harborseal_part1_annotated.csv'), str(BASE_DIR / 'annotated/harborseal_part2_annotated.csv')],
    'orca': [str(BASE_DIR / 'annotated/orca_part1_annotated.csv'), str(BASE_DIR / 'annotated/orca_part2_annotated.csv')],
    'sealion': [str(BASE_DIR / 'annotated/sealion_part1_annotated.csv'), str(BASE_DIR / 'annotated/sealion_part2_annotated.csv')],
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
                         'UMAP 1', 'UMAP 2', 'Cluster Label', 'Length']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols]
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
                html.P("Comparative Proteomics Visualization System",
                       style={'textAlign': 'center', 'color': '#64ffda', 'fontSize': 13,
                             'marginBottom': 8, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '3px', 'textTransform': 'uppercase'}),
                html.P("Select species for multi-dimensional protein space analysis",
                       style={'textAlign': 'center', 'color': '#a8dadc', 'fontSize': 16,
                             'marginBottom': 60, 'fontFamily': '"Inter", sans-serif',
                             'fontWeight': '300'}),
            ], style={'marginBottom': 20}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Img(src=animal_icons.get(species, ''),
                                style={'width': '120px', 'height': '120px', 'objectFit': 'contain',
                                      'marginBottom': 15,
                                      'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'}),
                        html.H3(info['display_name'],
                               style={'color': '#f1faee', 'marginBottom': 12, 'fontSize': 15,
                                     'fontFamily': '"Inter", sans-serif', 'fontWeight': '500'}),
                        html.Button('Select', id=f'btn-{species}', n_clicks=0,
                                   style={'padding': '10px 28px', 'fontSize': 11,
                                         'backgroundColor': 'transparent',
                                         'color': '#a8dadc',
                                         'border': '1px solid rgba(168, 218, 220, 0.3)',
                                         'borderRadius': 25,
                                         'cursor': 'pointer', 'transition': 'all 0.3s ease',
                                         'fontWeight': '500', 'fontFamily': '"JetBrains Mono", monospace',
                                         'letterSpacing': '2px', 'textTransform': 'uppercase'})
                    ], className='glass-card',
                       style={'padding': 25, 'textAlign': 'center',
                             'background': 'rgba(255, 255, 255, 0.03)',
                             'backdropFilter': 'blur(10px)',
                             'border': '1px solid rgba(255, 255, 255, 0.1)',
                             'borderRadius': 16},
                       id=f'card-{species}')
                ], style={'width': '18%', 'display': 'inline-block', 'margin': '1%',
                         'verticalAlign': 'top'})
                for species, info in SPECIES_DATA.items()
            ], style={'textAlign': 'center', 'marginBottom': 60}),

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
                        {'label': 'UNIQUE', 'value': 'unique'},
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

        # Full-width Plot
        html.Div([
            dcc.Graph(id='interactive-graph',
                     style={'height': 'calc(100vh - 200px)', 'width': '100%'},
                     config={'displayModeBar': True, 'displaylogo': False})
        ], style={'background': '#ffffff',
                 'borderRadius': 16, 'padding': 10,
                 'boxShadow': '0 10px 40px rgba(0, 0, 0, 0.2)',
                 'border': '1px solid rgba(100, 255, 218, 0.2)'}),

        # Protein Selection Table - Below the plot (full width)
        html.Div([
            html.Div([
                html.H3("Selected Proteins",
                       style={'color': '#64ffda', 'marginBottom': 8,
                             'fontSize': 12, 'fontFamily': '"JetBrains Mono", monospace',
                             'letterSpacing': '2px', 'textTransform': 'uppercase',
                             'fontWeight': '500', 'display': 'inline-block'}),
                html.Span(" — Use lasso or click to select proteins from the plot",
                         style={'fontSize': 12, 'color': '#a8dadc',
                               'fontFamily': '"JetBrains Mono", monospace',
                               'letterSpacing': '0.5px'})
            ], style={'marginBottom': 15}),
            html.Div(id='selection-table-container',
                    children="No proteins selected. Click or drag to select.",
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

    # Chat widget - floating toggle + panel
    html.Button('AI', id='chat-toggle-btn', n_clicks=0, className='chat-toggle'),
    html.Div([
        html.Div("Your Lab Guide", className='chat-header'),
        html.Div(
            id='chat-messages-display',
            children=[
                html.Div(dcc.Markdown(
                    "Hey there! \U0001F30A Welcome to the **Proteome Explorer**! "
                    "Each dot on this plot is a *protein* — and when dots **cluster together**, "
                    "it means those proteins share similar structure or function. Pretty neat, right?",
                    className='chat-md'),
                    className='chat-msg chat-msg-assistant'),
                html.Div(dcc.Markdown(
                    "Here's a fun thing to try: grab the **lasso tool** \U0001F3AF from the "
                    "toolbar (top-left of the plot) and draw around a cluster. "
                    "You'll see the protein details pop up in the table below!",
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
    """Compute unique/shared tags based on cluster membership across active species."""
    cluster_species = {}
    for sd in species_data_list:
        for rec in sd['df']:
            cl = rec.get('Cluster Label')
            if cl is not None and str(cl) != 'nan':
                cluster_species.setdefault(int(cl), set()).add(sd['name'])
    for sd in species_data_list:
        for rec in sd['df']:
            cl = rec.get('Cluster Label')
            if cl is not None and str(cl) != 'nan' and int(cl) in cluster_species:
                rec['filter_tag'] = 'unique' if len(cluster_species[int(cl)]) == 1 else 'shared'
            else:
                rec['filter_tag'] = 'shared'
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
     Input('active-species-store', 'data')]
)
def update_interactive_graph(point_size, opacity, loaded_data, protein_filter, active_species):
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
            if 'Cluster Label' in record:
                text += f"Cluster: {record['Cluster Label']}"
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
            customdata=[[r['Entry']] for r in df_records]
        ))
    margin = 50
    fig.update_layout(
        title=None,
        xaxis=dict(range=[-300 - margin, 300 + margin], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-300 - margin, 300 + margin], showgrid=False, zeroline=False, visible=False,
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

@app.callback(
    Output('selection-table-container', 'children'),
    [Input('interactive-graph', 'selectedData')],
    [State('loaded-data', 'data'),
     State('active-species-store', 'data')]
)
def show_selected_proteins(selectedData, loaded_data, active_species):
    if not selectedData or not loaded_data:
        return "No proteins selected. Click or drag to select."
    try:
        active_species = active_species or []
        # Filter to active species (same order as graph traces)
        active_data = [sd for sd in loaded_data if sd['name'] in active_species]

        # Build lookup: {species_name: {entry_id: record}} for customdata-based lookup
        species_lookup = {}
        for species_data in active_data:
            entry_map = {}
            for record in species_data['df']:
                entry_map[record['Entry']] = record
            species_lookup[species_data['name']] = entry_map

        selected_proteins = []
        for point in selectedData['points']:
            curve_num = point['curveNumber']
            if curve_num >= len(active_data):
                continue

            species_data = active_data[curve_num]

            # Use customdata to find the correct protein (filter-safe)
            customdata = point.get('customdata')
            if customdata and len(customdata) > 0:
                entry_id = customdata[0]
                protein = species_lookup.get(species_data['name'], {}).get(entry_id)
            else:
                continue

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
                'Organism': SPECIES_DATA[species_data['name']]['display_name'],
                'Protein Name': protein_name[:80],
                'Sequence': seq_preview,
                'Cluster': str(protein.get('Cluster Label', 'N/A')),
                'Domains': ann.get('hmm_labels', '') or '\u2014' if ann else '\u2014',
                'HMM Descriptions': ann.get('hmm_descriptions', '') or '\u2014' if ann else '\u2014',
                'Pfam IDs': ann.get('hmm_pfam_ids', '') or '\u2014' if ann else '\u2014',
                'Domain Ranges': ann.get('hmm_ranges', '') or '\u2014' if ann else '\u2014',
            })
        return html.Div([
            html.Div(f"{len(selected_proteins)} protein{'s' if len(selected_proteins) != 1 else ''} selected",
                    style={'fontSize': 13, 'fontWeight': '600', 'marginBottom': 15,
                          'color': '#64ffda', 'fontFamily': '"JetBrains Mono", monospace',
                          'letterSpacing': '1px'}),
            dash_table.DataTable(
                data=selected_proteins,
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
                    'textAlign': 'left',
                    'padding': '12px 10px',
                    'fontFamily': '"JetBrains Mono", monospace',
                    'fontSize': 11,
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': '180px',
                    'backgroundColor': 'transparent',
                    'border': 'none'
                },
                style_header={
                    'backgroundColor': 'rgba(100, 255, 218, 0.1)',
                    'fontWeight': '600',
                    'color': '#64ffda',
                    'borderBottom': '1px solid rgba(100, 255, 218, 0.2)',
                    'fontSize': 10,
                    'textTransform': 'uppercase',
                    'letterSpacing': '1px'
                },
                style_data={
                    'backgroundColor': 'rgba(10, 25, 47, 0.4)',
                    'color': '#f1faee',
                    'borderBottom': '1px solid rgba(100, 255, 218, 0.1)',
                    'cursor': 'pointer'
                },
                css=[{
                    'selector': 'td.dash-cell',
                    'rule': 'color: #f1faee !important;'
                }, {
                    'selector': 'td.dash-cell.focused',
                    'rule': 'background-color: rgba(100, 255, 218, 0.15) !important; color: #f1faee !important; outline: none !important;'
                }, {
                    'selector': 'input.dash-cell-value',
                    'rule': 'background-color: transparent !important; color: #f1faee !important; caret-color: transparent !important; border: none !important;'
                }, {
                    'selector': 'tr:hover td',
                    'rule': 'color: #f1faee !important;'
                }, {
                    'selector': '.dash-spreadsheet-inner tr:hover td::after, .dash-spreadsheet-inner td::after',
                    'rule': 'display: none !important;'
                }, {
                    'selector': '.dash-spreadsheet-inner tr:hover',
                    'rule': 'background: none !important;'
                }],
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(10, 25, 47, 0.6)'},
                    {'if': {'state': 'active'}, 'backgroundColor': 'rgba(100, 255, 218, 0.15)',
                     'color': '#f1faee',
                     'border': '1px solid rgba(100, 255, 218, 0.3)',
                     'overflow': 'visible',
                     'whiteSpace': 'normal',
                     'height': 'auto'}
                ],
                editable=False,
                page_size=10,
                page_action='native',
                sort_action='native',
                sort_by=[{'column_id': 'Protein Name', 'direction': 'asc'}]
            )
        ])
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


# Server callback: process query when pending-query is set
@app.callback(
    [Output('chat-messages-display', 'children'),
     Output('chat-history', 'data')],
    Input('pending-query', 'data'),
    [State('chat-history', 'data'),
     State('active-species-store', 'data'),
     State('interactive-graph', 'selectedData'),
     State('loaded-data', 'data')],
    prevent_initial_call=True,
)
def process_chat_query(pending_query, chat_history, active_species, selected_data, loaded_data):
    if not pending_query:
        raise dash.exceptions.PreventUpdate

    chat_history = chat_history or []
    visual_context = build_visual_context(active_species, selected_data, loaded_data)

    mode, text = route_message(chat_history, pending_query, visual_context)
    if mode == "rag":
        raw = query_rag(text)
        if raw.startswith("Error:") or raw.startswith("The knowledge base service"):
            # RAG unavailable — fall back to answering from visual context or chat
            if visual_context:
                answer = answer_with_context(text, visual_context, chat_history)
            else:
                answer = chat_reply(pending_query, chat_history, visual_context)
        else:
            answer = rewrite_as_guide(raw, pending_query, chat_history, visual_context)
    elif mode == "visual":
        answer = answer_with_context(text, visual_context, chat_history)
    else:
        answer = chat_reply(pending_query, chat_history, visual_context)

    chat_history.append({"role": "user", "content": pending_query})
    chat_history.append({"role": "assistant", "content": answer})

    message_divs = []
    for msg in chat_history:
        if msg['role'] == 'user':
            message_divs.append(html.Div(msg['content'], className='chat-msg chat-msg-user'))
        else:
            message_divs.append(html.Div(
                dcc.Markdown(msg['content'], className='chat-md'),
                className='chat-msg chat-msg-assistant'
            ))

    return message_divs, chat_history


server = app.server

if __name__ == '__main__':
    import os
    print("\n" + "="*60)
    print("Marine Mammal Proteome Explorer - Ocean Edition")
    print("="*60)
    _load_vectorstore()
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
