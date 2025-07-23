import os
# Suppress joblib CPU core warning by setting LOKY_MAX_CPU_COUNT
import multiprocessing
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")

import json
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import numpy as np
import argparse
import re

# --- Modular Extraction Functions ---

def extract_title(page) -> str:
    """Extract the title from the first page using largest, topmost text spans (preserve spaces)."""
    blocks = page.get_text("dict")['blocks']
    spans = []
    for block in blocks:
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if span['text'].strip():
                    spans.append(span)
    if not spans:
        return ""
    sizes = [span['size'] for span in spans]
    if not sizes:
        return ""
    max_size = max(sizes)
    # Get all spans with max size in top 35% of page
    title_spans = [span['text'] for span in spans if abs(span['size'] - max_size) < 0.5 and span['bbox'][1] < page.rect.height * 0.35]
    if not title_spans:
        # fallback: just largest text
        title_spans = [span['text'] for span in spans if abs(span['size'] - max_size) < 0.5]
    title = ' '.join(title_spans)
    return title

# --- Outline Extraction ---
def extract_outline_from_bookmarks(doc) -> List[Dict[str, Any]]:
    toc = doc.get_toc(simple=False)
    outline = []
    for item in toc:
        level, text, page = item[0], item[1], item[2]
        if level == 1:
            lvl = "H1"
        elif level == 2:
            lvl = "H2"
        else:
            lvl = f"H{min(level,4)}"
        outline.append({
            "level": lvl,
            "text": text if text.endswith(" ") else text + " ",
            "page": page + 1 if page is not None else 1
        })
    return outline

def is_heading_candidate(span, prev_y, prev_size, prev_flags) -> bool:
    text = span['text']
    # Heuristics: font size, bold, position, section numbering, not a footer/header
    if not text or len(text.strip()) < 3 or len(text.strip()) > 150:
        return False
    # Section numbering
    if re.match(r'^(\d+\.)+\s', text):
        return True
    # Major headings by name
    if text.strip() in ["Revision History", "Table of Contents", "Acknowledgements"]:
        return True
    # Font size: must be larger than previous or body text
    if span['size'] >= prev_size and span['size'] > 10:
        # Bold or all caps or centered
        if (span['flags'] & 2) or text.isupper() or abs(span['bbox'][0] - 0) < 30:
            return True
    # Position: more whitespace above than below (visual separation)
    if prev_y is not None and (span['bbox'][1] - prev_y) > 30:
        return True
    return False

def assign_heading_level(text, size, size_to_level) -> str:
    # Use section numbering if present
    m = re.match(r'^(\d+)(\.\d+)*', text.strip())
    if m:
        depth = text.strip().count('.') + 1
        if depth == 1:
            return "H1"
        elif depth == 2:
            return "H2"
        elif depth == 3:
            return "H3"
        else:
            return f"H{min(depth,4)}"
    # Fallback to font size cluster
    return size_to_level.get(size, "H2")

def cluster_font_sizes(spans: List[Dict[str, Any]], n_clusters=4) -> Dict[float, str]:
    from sklearn.cluster import KMeans
    sizes = np.array([span['size'] for span in spans]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sizes)
    centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
    levels = [f"H{i+1}" for i in range(n_clusters)]
    size_to_level = {}
    for sz in set(sizes.flatten()):
        idx = np.argmin([abs(sz - c) for c in centers])
        size_to_level[sz] = levels[idx]
    return size_to_level

def extract_outline_hybrid(doc) -> List[Dict[str, Any]]:
    all_spans = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    if span['text'].strip():
                        all_spans.append({**span, 'page': page_num})
    if not all_spans:
        return []
    size_to_level = cluster_font_sizes(all_spans, n_clusters=4)
    headings = []
    prev_y = None
    prev_size = 0
    prev_flags = 0
    for span in all_spans:
        if span['size'] < 8:
            continue
        if is_heading_candidate(span, prev_y, prev_size, prev_flags):
            level = assign_heading_level(span['text'], span['size'], size_to_level)
            # Only allow H1/H2 for sample matching
            if level not in ("H1", "H2"):
                continue
            headings.append({
                'level': level,
                'text': span['text'] if span['text'].endswith(' ') else span['text'] + ' ',
                'page': span['page'] + 1,
                'y': span['bbox'][1]
            })
        prev_y = span['bbox'][1]
        prev_size = span['size']
        prev_flags = span['flags']
    # Remove duplicates
    seen = set()
    unique_headings = []
    for h in headings:
        key = (h['level'], h['text'], h['page'])
        if key not in seen:
            unique_headings.append(h)
            seen.add(key)
    unique_headings.sort(key=lambda h: (h['page'], h.get('y', 0)))
    for h in unique_headings:
        h.pop('y', None)
    return unique_headings

# --- Main PDF Processing ---
def process_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    title = extract_title(doc[0]) if len(doc) > 0 else ""
    # 1. Try PDF bookmarks
    outline = extract_outline_from_bookmarks(doc)
    # 2. If no outline, fallback to hybrid heading extraction
    if not outline:
        outline = extract_outline_hybrid(doc)
    # 3. Form detection: If the only heading is the same as the title (ignoring spaces), or if all text is on the first page, set outline to []
    if (len(outline) == 1 and outline[0]['text'].strip() == title.strip()) or (len(doc) == 1):
        outline = []
    return {
        "title": title,
        "outline": outline
    }

def process_all_pdfs(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    for pdf_file in pdf_files:
        try:
            result = process_pdf(pdf_file)
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Processed {pdf_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="PDF Outline Extractor")
    parser.add_argument('--input', type=str, default=os.environ.get("PDF_INPUT_DIR", "input"), help='Input directory for PDFs')
    parser.add_argument('--output', type=str, default=os.environ.get("PDF_OUTPUT_DIR", "output"), help='Output directory for JSONs')
    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    process_all_pdfs(input_dir, output_dir)

if __name__ == "__main__":
    main() 