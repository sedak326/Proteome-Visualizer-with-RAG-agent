#!/usr/bin/env python3
"""
extract_corpus.py — Structured Knowledge Base Extractor

Replaces the old extract_image_summaries.py + augment_corpus.py + AutoRAG parse/chunk pipeline.
Produces a single corpus.jsonl with structured metadata (document, chapter, section, page)
preserved in every chunk.

Sources:
  - data/Ch*.html  → HTML ebook chapters (book of marine-mammal species accounts)
  - data/*.pdf     → Research papers (PDF)

Output:
  project/corpus.jsonl

Run from /home/skavlak/AI_Agent/ with the autorag conda env:
  conda run -n autorag python extract_corpus.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
from bs4 import BeautifulSoup, Comment, NavigableString
from openai import OpenAI
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


MONO_HINTS = ("courier", "consolas", "menlo", "monaco", "dejavusansmono", "sourcecode", "code", "mono")

CHAPTER_PATTERN = re.compile(
    r"^(Chapter\s+\d+|Section\s+\d+|\d+(\.\d+)*\.?\s+\S)",
    re.IGNORECASE,
)

MAX_SECTION_CHARS = 4000  # max chars per prose chunk before splitting

MIN_IMG_DIM = 150  # skip images smaller than this in either dimension (filters junk sub-images)

BOLD_FLAG = 16  # PyMuPDF span flag bit for bold text

# Stop chunking text when one of these section headings is detected in a PDF
STOP_SECTION_RE = re.compile(
    r"^(References|Bibliography|Acknowledgements?|Funding|Supplementary)",
    re.IGNORECASE,
)

# Skip text blocks that are figure or table captions
CAPTION_RE = re.compile(r"^(Fig(ure)?\.?\s*\d+|Table\s+\d+[\.\:\s])", re.IGNORECASE)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_monospace_font(font_name: str) -> bool:
    fn = (font_name or "").lower().replace(" ", "")
    return any(h in fn for h in MONO_HINTS)


def normalize_ws_keep_indent(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def is_heading(text: str, size: float, median_size: float, flags: int = 0) -> bool:
    is_bold = bool(flags & BOLD_FLAG)
    # Scientific papers use bold for section headings; size alone is unreliable
    # (two-column layouts keep uniform body size for headings like Introduction/Methods).
    # Fall back to size-only for clearly oversized text (e.g. paper title on first page).
    is_clearly_large = size >= median_size * 1.4
    if not is_bold and not is_clearly_large:
        return False
    # Reject tiny bold — inline emphasis (**Note:**, **p** < 0.05, etc.)
    if is_bold and size < median_size * 0.9:
        return False
    t = text.strip()
    if len(t) < 3 or len(t) > 120:
        return False
    if re.fullmatch(r"[\d\W]+", t):
        return False
    return True


def parse_chapter_from_heading(heading: str) -> str:
    m = re.match(r"^(\d+)\.", heading.strip())
    if m:
        return f"Chapter {m.group(1)}"
    m2 = re.match(r"^Chapter\s+(\d+)", heading.strip(), re.IGNORECASE)
    if m2:
        return f"Chapter {m2.group(1)}"
    return "Unknown Chapter"


@dataclass
class TextSpan:
    text: str
    size: float
    font: str
    flags: int


def extract_spans(page: fitz.Page) -> List[TextSpan]:
    spans = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                t = sp.get("text", "").strip()
                if t:
                    spans.append(TextSpan(
                        text=t,
                        size=float(sp.get("size") or 0),
                        font=str(sp.get("font") or ""),
                        flags=int(sp.get("flags") or 0),
                    ))
    return spans


def guess_headings_for_page(page: fitz.Page) -> List[str]:
    spans = extract_spans(page)
    if not spans:
        return []
    sizes = sorted(s.size for s in spans if s.size > 0)
    if not sizes:
        return []
    median = sizes[len(sizes) // 2]
    seen = set()
    out = []
    for s in spans:
        if is_heading(s.text, s.size, median, s.flags):
            key = s.text.strip().lower()
            if key not in seen:
                seen.add(key)
                out.append(s.text.strip())
    return out



def extract_text_blocks_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    # First pass: collect all span sizes across the document to get a stable median
    all_sizes: List[float] = []
    for page_index in range(len(doc)):
        for block in doc[page_index].get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    s = float(sp.get("size") or 0)
                    if s > 0:
                        all_sizes.append(s)
    all_sizes.sort()
    doc_median = all_sizes[len(all_sizes) // 2] if all_sizes else 10.0

    out = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        for b_i, block in enumerate(page.get_text("dict").get("blocks", [])):
            if block.get("type") != 0:
                continue
            pieces = []
            mono_chars = 0
            total_chars = 0
            # Track dominant bold/size for heading detection
            bold_chars = 0
            size_sum = 0.0
            size_count = 0
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    t = sp.get("text", "")
                    if not t:
                        continue
                    pieces.append(t)
                    font = str(sp.get("font") or "")
                    total_chars += len(t)
                    if is_monospace_font(font):
                        mono_chars += len(t)
                    if int(sp.get("flags") or 0) & BOLD_FLAG:
                        bold_chars += len(t)
                    sz = float(sp.get("size") or 0)
                    if sz > 0:
                        size_sum += sz * len(t)
                        size_count += len(t)
            text = normalize_ws_keep_indent("".join(pieces))
            if not text.strip():
                continue
            mono_ratio = (mono_chars / total_chars) if total_chars else 0.0
            avg_size = (size_sum / size_count) if size_count else 0.0
            # Dominant flags: treat block as bold if >60 % of chars are bold
            dom_flags = BOLD_FLAG if (total_chars and bold_chars / total_chars > 0.6) else 0
            bbox = block.get("bbox", (0, 0, 0, 0))
            out.append({
                "page_number": page_index + 1,
                "element_id": f"p{page_index+1}_t{b_i}",
                "text": text,
                "mono_ratio": round(mono_ratio, 3),
                "bbox_y0": bbox[1],
                "is_heading": is_heading(text.strip(), avg_size, doc_median, dom_flags),
            })
    doc.close()
    return out


def extract_tables_pdfplumber(pdf_path: Path) -> List[Dict[str, Any]]:
    out = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for t_i, table in enumerate(tables):
                if not table:
                    continue
                rows = [[(c or "").strip() for c in row] for row in table]
                rows = [r for r in rows if any(cell for cell in r)]
                if not rows:
                    continue
                tsv = "\n".join("\t".join(r).rstrip() for r in rows).strip()
                if tsv:
                    out.append({
                        "page_number": page_index + 1,
                        "element_id": f"p{page_index+1}_tbl{t_i}",
                        "text": tsv,
                        "rows": rows,
                    })
    return out


def extract_images_pymupdf(pdf_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(str(pdf_path))
    img_dir = out_dir / "images"
    ensure_dir(img_dir)
    out = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        # Build xref → bbox map from image blocks so we can sort images spatially
        xref_bbox: Dict[int, tuple] = {}
        for blk in page.get_text("dict").get("blocks", []):
            if blk.get("type") == 1:  # image block
                xref_bbox[blk.get("image", 0)] = blk.get("bbox", (0, 0, 0, 0))

        for img_i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                # Filter out junk sub-images (colour bands, tiny decorations)
                if pix.width < MIN_IMG_DIM or pix.height < MIN_IMG_DIM:
                    pix = None
                    continue
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_name = f"page_{page_index+1:03d}_img_{img_i:03d}.png"
                img_path = img_dir / img_name
                pix.save(str(img_path))
                pix = None
            except Exception:
                continue
            bbox = xref_bbox.get(xref, (0, 0, 0, 0))
            out.append({
                "page_number": page_index + 1,
                "element_id": f"p{page_index+1}_img{img_i}",
                "image_name": img_name,
                "image_path": str(img_path),
                "bbox_y0": bbox[1],
            })
    doc.close()
    return out

# this checks if a block is monospaced, I had this in here for a different use case, 
# but since most scientific papers about marine mammals don't include code blocks, it's not really needed 
def block_is_code(text: str, mono_ratio: float) -> bool:
    if mono_ratio >= 0.6:
        return True
    if re.match(r"^(GET|POST|PUT|DELETE|PATCH)\s+https?://", text.strip()):
        return True
    stripped = text.strip()
    if (stripped.startswith("{") or stripped.startswith("[")) and (
        stripped.endswith("}") or stripped.endswith("]")
    ):
        return True
    if stripped.startswith("curl "):
        return True
    return False


def format_block_with_code(text: str, mono_ratio: float) -> str:
    if block_is_code(text, mono_ratio):
        return f"```\n{text}\n```"
    return text


def build_docs(
    document_name: str,
    text_elements: List[Dict[str, Any]],
    table_elements: List[Dict[str, Any]],
    image_elements: List[Dict[str, Any]],
    headings_by_page: Dict[int, List[str]],
    do_ocr: bool = False,
    summarize_images: bool = False,
    image_summarizer_fn: Optional[Callable[[str, str, str], str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build indexable chunks preserving document_name, chapter, section, page.

    image_summarizer_fn(image_path, document_name, section_title) -> str
        Called for each image when summarize_images=True and the callable is provided.
    """
    docs: List[Dict[str, Any]] = []

    current_section = "Introduction"
    current_chapter = "Chapter 1"
    chunk_buf: List[str] = []
    chunk_pages: List[int] = []
    chunk_counter = 0

    def flush_chunk() -> None:
        nonlocal chunk_buf, chunk_pages, chunk_counter
        if not chunk_buf:
            return
        text = "\n\n".join(chunk_buf).strip()
        if not text:
            chunk_buf = []
            chunk_pages = []
            return
        chunk_counter += 1
        page_start = min(chunk_pages) if chunk_pages else None
        page_end = max(chunk_pages) if chunk_pages else None
        page_ref = page_start if page_start == page_end else f"{page_start}-{page_end}"
        docs.append({
            "content_type": "text",
            "id": f"chunk_{chunk_counter:04d}",
            "document_name": document_name,
            "chapter": current_chapter,
            "section_title": current_section,
            "page_number": page_start,
            "page_range": str(page_ref),
            "text": (
                f"[Document: {document_name} | Chapter: {current_chapter} | "
                f"Section: {current_section} | Page: {page_ref}]\n\n{text}"
            ),
        })
        chunk_buf = []
        chunk_pages = []

    def elem_sort_key(e: Dict[str, Any]) -> tuple:
        # Sort by page first, then by vertical position on the page (y0)
        return (int(e.get("page_number") or 0), float(e.get("bbox_y0") or 0))

    # Pre-compute text frequency to detect running headers/footers
    text_freq = Counter(e["text"].strip() for e in text_elements)

    # Snapshot of current_section after processing each page (used for image assignment)
    section_at_page: Dict[int, str] = {}

    stop_text = False
    for e in sorted(text_elements, key=elem_sort_key):
        if stop_text:
            break
        page = int(e.get("page_number") or 0)
        raw = (e.get("text") or "").strip()
        if not raw:
            continue

        # Skip running headers/footers (short text repeated across many pages)
        if text_freq[raw] > 3 and len(raw) < 200:
            continue

        # Skip figure and table captions (image chunks carry GPT summaries instead)
        if CAPTION_RE.match(raw):
            continue

        # Inline heading detection (bold-primary; fires mid-page unlike the page map)
        if e.get("is_heading"):
            if STOP_SECTION_RE.match(raw):
                flush_chunk()
                stop_text = True
                break
            if raw != current_section:
                flush_chunk()
                current_section = raw
                current_chapter = parse_chapter_from_heading(current_section)
            section_at_page[page] = current_section
            continue  # don't add the heading text itself as body content

        mono_ratio = e.get("mono_ratio", 0.0)
        formatted = format_block_with_code(raw, mono_ratio)
        current_len = sum(len(b) for b in chunk_buf)
        if current_len + len(formatted) > MAX_SECTION_CHARS and chunk_buf:
            flush_chunk()
        chunk_buf.append(formatted)
        chunk_pages.append(page)
        section_at_page[page] = current_section

    flush_chunk()

    # --- Tables ---
    for t in sorted(table_elements, key=elem_sort_key):
        page = int(t.get("page_number") or 0)
        headings = headings_by_page.get(page, [])
        sec = headings[0] if headings else current_section
        chap = parse_chapter_from_heading(sec)
        docs.append({
            "content_type": "table",
            "id": t.get("element_id"),
            "document_name": document_name,
            "chapter": chap,
            "section_title": sec,
            "page_number": page,
            "page_range": str(page),
            "text": (
                f"[Document: {document_name} | Chapter: {chap} | "
                f"Section: {sec} | Page: {page}]\n\n[TABLE]\n{t.get('text', '')}"
            ),
        })

    total_images = len(image_elements)
    for idx, im in enumerate(sorted(image_elements, key=elem_sort_key)):
        page = int(im.get("page_number") or 0)
        # Use the section that was active when text on this page was processed,
        # falling back to the previous page if no text was found on this page.
        sec = (
            section_at_page.get(page)
            or section_at_page.get(page - 1)
            or current_section
        )
        chap = parse_chapter_from_heading(sec)
        img_path = im.get("image_path", "")
        img_name = im.get("image_name", "")

        log.info("  Image %d/%d: %s", idx + 1, total_images, img_name)

        description = ""
        if summarize_images and image_summarizer_fn is not None:
            try:
                description = image_summarizer_fn(img_path, document_name, sec)
                if description:
                    log.info("    Summary: %s...", description[:80])
            except Exception as exc:
                log.warning("    Image summarization failed for %s: %s", img_path, exc)

        body = f"[IMAGE: {img_name}]"
        if description:
            body += f"\n\n{description}"
        else:
            body += "\n\n[No description available]"

        docs.append({
            "content_type": "image",
            "id": im.get("element_id"),
            "document_name": document_name,
            "chapter": chap,
            "section_title": sec,
            "page_number": page,
            "page_range": str(page),
            "image_name": img_name,
            "image_path": img_path,
            "text": (
                f"[Document: {document_name} | Chapter: {chap} | "
                f"Section: {sec} | Page: {page}]\n\n{body}"
            ),
        })

    return docs

def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _image_to_jpeg_b64(image_path: str) -> str:
    """Load a PNG/JPEG and return base64-encoded JPEG string."""
    with open(image_path, "rb") as f:
        raw = f.read()
    img = Image.open(io.BytesIO(raw))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def summarize_image_gpt4o(
    image_path: str,
    client: OpenAI,
    document_name: str,
    section_title: str,
) -> str:
    """
    Describe an image using GPT-4o-mini vision.
    Returns a plain-text description suitable for embedding.
    """
    b64 = _image_to_jpeg_b64(image_path)
    prompt = (
        f"You are analyzing a figure from a marine biology research paper titled '{document_name}', "
        f"in the section '{section_title}'. "
        "Describe what this figure shows in detail: phylogenetic trees and species/clades, "
        "protein expression heatmaps or bar plots (axes, values, comparisons), genome maps or "
        "synteny plots, population genetics charts, and any quantitative data visible. "
        "Your description will be used to make this figure searchable in a RAG system. "
        "Be specific. Do not say 'the image shows' — just describe it directly."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def process_pdf(pdf_path: Path, out_dir: Path, client: OpenAI) -> List[Dict[str, Any]]:
    """
    Extract all chunks (text, tables, images) from a PDF file.
    Images are summarized via GPT-4o-mini; failures are logged and skipped.
    Returns a list of chunk dicts ready for corpus.jsonl.
    """
    document_name = pdf_path.stem.replace("_", " ").replace("-", " ")
    # Title-case only if it's all lowercase/uppercase
    if document_name == document_name.lower() or document_name == document_name.upper():
        document_name = document_name.title()

    log.info("\nPDF: %s  →  '%s'", pdf_path.name, document_name)

    log.info("  Extracting text blocks…")
    text_elements = extract_text_blocks_pymupdf(pdf_path)
    log.info("    %d text blocks", len(text_elements))

    log.info("  Extracting tables…")
    table_elements = extract_tables_pdfplumber(pdf_path)
    log.info("    %d tables", len(table_elements))

    log.info("  Extracting images…")
    image_elements = extract_images_pymupdf(pdf_path, out_dir)
    log.info("    %d images", len(image_elements))

    log.info("  Detecting headings…")
    fitz_doc = fitz.open(str(pdf_path))
    headings_by_page: Dict[int, List[str]] = {}
    for i in range(len(fitz_doc)):
        headings_by_page[i + 1] = guess_headings_for_page(fitz_doc[i])
    fitz_doc.close()

    # Build the GPT summarizer callback bound to this document's client
    def _summarize(img_path: str, doc_name: str, section: str) -> str:
        return summarize_image_gpt4o(img_path, client, doc_name, section)

    log.info("  Building chunks…")
    docs = build_docs(
        document_name=document_name,
        text_elements=text_elements,
        table_elements=table_elements,
        image_elements=image_elements,
        headings_by_page=headings_by_page,
        do_ocr=False,
        summarize_images=True,
        image_summarizer_fn=_summarize,
    )

    # Make IDs unique across multiple PDFs by prefixing with a short document slug
    slug = re.sub(r"[^a-z0-9]+", "_", document_name.lower())[:30].strip("_")
    for doc in docs:
        if doc.get("id"):
            doc["id"] = f"pdf_{slug}_{doc['id']}"

    counts = Counter(d["content_type"] for d in docs)
    log.info("  Chunks: %s", dict(counts))
    return docs


def _get_p_text_clean(p_tag) -> str:
    """
    Extract visible text from a <p> tag, excluding epub:type=pagebreak subtrees
    (those contain only artefact text like 'Page 2').

    Note: BeautifulSoup NavigableString objects have name=None (not absent),
    so we must check `node.name is None` rather than `hasattr(node, 'name')`.
    """
    parts = []
    for node in p_tag.descendants:
        # Skip Tag nodes (name is a non-None string); keep only plain NavigableStrings.
        # Also skip Comment nodes (subclass of NavigableString) — they contain
        # commented-out HTML markup that looks like raw HTML in the text.
        if getattr(node, "name", None) is not None:
            continue
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue
        # Walk ancestors up to p_tag to check whether this string should be skipped:
        # - epub:type=pagebreak subtrees contain only "Page N" artefact text
        # - role="button" / data-dfn spans are epub tooltip wrappers that duplicate
        #   the citation text already present as a sibling NavigableString
        parent = node.parent
        in_pb = False
        while parent and parent is not p_tag:
            attrs = getattr(parent, "attrs", {})
            if (attrs.get("epub:type") == "pagebreak"
                    or attrs.get("data-dfn")
                    or attrs.get("role") == "button"):
                in_pb = True
                break
            parent = parent.parent
        if not in_pb:
            s = str(node).strip()
            if s:
                parts.append(s)
    return " ".join(parts)


def process_html(html_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a single HTML chapter file and return structured text chunks.

    Filename convention: Ch075_274-277_<isbn>.html
      → chapter = "Chapter 75", page_range = "274-277", start_page = 274
    Also handles pages with a leading P: Ch013_P42-44_<isbn>.html
    """
    stem = html_path.stem  # e.g. "Ch075_274-277_B978..."

    m = re.match(r"Ch(\d+)_P?(\d+)-(\d+)", stem)
    if not m:
        log.warning("Skipping %s — filename doesn't match expected pattern", html_path.name)
        return []

    chapter_num = int(m.group(1))
    start_page = int(m.group(2))
    end_page = int(m.group(3))
    chapter = f"Chapter {chapter_num}"
    page_range_str = f"{start_page}-{end_page}"

    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Document name from <h1>
    h1 = soup.find("h1")
    document_name = h1.get_text(strip=True) if h1 else html_path.stem

    # State
    current_section = "Introduction"
    current_page = start_page
    chunk_buf: List[str] = []
    docs: List[Dict[str, Any]] = []
    chunk_counter = 0

    def flush_chunk() -> None:
        nonlocal chunk_buf, chunk_counter
        if not chunk_buf:
            return
        text = "\n\n".join(chunk_buf).strip()
        if not text:
            chunk_buf.clear()
            return
        chunk_counter += 1
        cid = f"html_ch{chapter_num:03d}_chunk_{chunk_counter:04d}"
        docs.append({
            "content_type": "text",
            "id": cid,
            "document_name": document_name,
            "chapter": chapter,
            "section_title": current_section,
            "page_number": current_page,
            "page_range": page_range_str,
            "text": (
                f"[Document: {document_name} | Chapter: {chapter} | "
                f"Section: {current_section} | Page: {current_page}]\n\n{text}"
            ),
        })
        chunk_buf.clear()

    # Walk all elements in document order.
    # Track processed element IDs to avoid double-visiting
    processed_ids: set = set()
    done = False

    for elem in soup.find_all(True):
        if done:
            break
        if id(elem) in processed_ids:
            continue

        epub_type = (elem.attrs or {}).get("epub:type", "")

        # Pagebreak (standalone, not inside a <p> we've already handled) 
        if epub_type == "pagebreak":
            label = (elem.attrs or {}).get("aria-label", "")
            pg_m = re.search(r"\d+", label)
            if pg_m:
                current_page = int(pg_m.group())
            processed_ids.add(id(elem))
            continue

        # Section headings
        if elem.name == "h2":
            cls = elem.get("class") or []
            if "reftitle" in cls:
                done = True
                break
            if "h1hd" in cls:
                flush_chunk()
                current_section = elem.get_text(strip=True)
            processed_ids.add(id(elem))
            for desc in elem.descendants:
                if hasattr(desc, "name"):
                    processed_ids.add(id(desc))
            continue

        # Figures — skip entirely
        if elem.name == "figure":
            processed_ids.add(id(elem))
            for desc in elem.descendants:
                if hasattr(desc, "name"):
                    processed_ids.add(id(desc))
            continue

        # Paragraphs 
        if elem.name == "p":
            # Handle inline pagebreak spans first
            for pb in elem.find_all(attrs={"epub:type": "pagebreak"}):
                label = pb.get("aria-label", "")
                pg_m = re.search(r"\d+", label)
                if pg_m:
                    current_page = int(pg_m.group())

            text = _get_p_text_clean(elem)
            if text:
                chunk_buf.append(text)
                if sum(len(b) for b in chunk_buf) > MAX_SECTION_CHARS:
                    flush_chunk()

            processed_ids.add(id(elem))
            for desc in elem.descendants:
                if hasattr(desc, "name"):
                    processed_ids.add(id(desc))
            continue

    flush_chunk()
    return docs


def main() -> None:
    data_dir = Path("data")
    out_dir = Path("project")
    ensure_dir(out_dir)

    all_docs: List[Dict[str, Any]] = []

    # Finds all html chapters starting with "Ch"
    html_files = sorted(data_dir.glob("Ch*.html"))
    log.info("Found %d HTML chapter files", len(html_files))
    for html_path in html_files:
        chunks = process_html(html_path)
        log.info("  %s → %d chunks", html_path.name, len(chunks))
        all_docs.extend(chunks)

    html_count = len(all_docs)

    client = OpenAI()
    img_out = out_dir / "extracted_images"
    ensure_dir(img_out)

    # Finds all PDf files
    pdf_files = sorted(data_dir.glob("*.pdf"))
    log.info("\nFound %d PDF files", len(pdf_files))
    for pdf_path in pdf_files:
        chunks = process_pdf(pdf_path, img_out, client)
        all_docs.extend(chunks)

    pdf_count = len(all_docs) - html_count

    # writes every chunk to corpus.jsonl
    out_path = out_dir / "corpus.jsonl"
    write_jsonl(out_path, all_docs)

    # prints a summary
    total = len(all_docs)
    type_counts = Counter(d["content_type"] for d in all_docs)
    print(f"\n{'='*60}")
    print(f"corpus.jsonl written → {out_path}")
    print(f"  Total chunks : {total}")
    print(f"  From HTML    : {html_count}")
    print(f"  From PDF     : {pdf_count}")
    print(f"  By type      : {dict(type_counts)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
