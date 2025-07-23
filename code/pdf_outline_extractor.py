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


def extract_title(page) -> str:
    """
    Extract the title from the first page: concatenate largest text spans that are close together vertically.
    """
    blocks = page.get_text("dict")['blocks']
    spans = []
    for block in blocks:
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                if span['text'].strip():
                    spans.append(span)
    if not spans:
        return ""
    # Find the largest font size
    max_size = max(span['size'] for span in spans)
    # Get all spans with the largest font size
    largest_spans = [span for span in spans if abs(span['size'] - max_size) < 0.5]
    # Sort by vertical position (y)
    largest_spans.sort(key=lambda s: s['bbox'][1])
    # Concatenate text if spans are close together vertically (within 30 units)
    title_lines = []
    prev_y = None
    for span in largest_spans:
        if prev_y is None or abs(span['bbox'][1] - prev_y) < 30:
            title_lines.append(span['text'].strip())
            prev_y = span['bbox'][1]
        else:
            break
    title = ' '.join(title_lines).strip()
    return title


def cluster_font_sizes(spans: List[Dict[str, Any]], n_clusters=4) -> Dict[float, str]:
    """
    Cluster font sizes to assign heading levels (H1, H2, H3, H4).
    Returns a mapping from font size to heading level.
    """
    sizes = np.array([span['size'] for span in spans]).reshape(-1, 1)
    unique_sizes = sorted(set(sizes.flatten()), reverse=True)
    if len(unique_sizes) <= n_clusters:
        levels = [f"H{i+1}" for i in range(len(unique_sizes))]
        return {sz: lvl for sz, lvl in zip(unique_sizes, levels)}
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sizes)
    centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
    levels = [f"H{i+1}" for i in range(n_clusters)]
    size_to_level = {}
    for i, c in enumerate(centers):
        # assign all sizes closest to this center
        for sz in set(sizes.flatten()):
            if abs(sz - c) < 0.1:
                size_to_level[sz] = levels[i]
    # fallback for any missing
    for sz in set(sizes.flatten()):
        if sz not in size_to_level:
            idx = np.argmin([abs(sz - c) for c in centers])
            size_to_level[sz] = levels[idx]
    return size_to_level




import re

def is_heading_candidate(text: str) -> bool:
    """
    Determines whether a given text span is likely a heading using general heuristics.
    Avoids hardcoding any document-specific terms.
    """

    clean = text.strip().rstrip(':').strip()

    # Reject empty or whitespace-only
    if not clean:
        return False

    # ❌ Reject lines that are obviously too short or too long
    if len(clean) > 100 or len(clean) < 5:
        return False

    word_count = len(clean.split())
    if word_count < 2 or word_count > 12:
        return False

    # ❌ Reject full sentences that end in a period
    if clean.endswith('.'):
        return False

    # ❌ Reject lines that are only numbers or symbols
    if re.fullmatch(r'[\W\d\s]+', clean):
        return False

    # ❌ Reject lines that look like full names or name lists
    if re.fullmatch(r'^([A-Z][a-z]+[\s,]*){1,5}$', clean):
        return False
    
        # ❌ Reject lines like 'Page X of Y'
    if re.match(r'Page\s+\d+\s+of\s+\d+', clean, re.IGNORECASE):
        return False

    # ❌ Reject version numbers like 'Version 1.0', 'Version 2022'
    if re.match(r'Version\s+\d+(\.\d+)*', clean, re.IGNORECASE):
        return False


    # ❌ Reject lines that are single years or look like dates
    if re.fullmatch(r'\d{4}', clean):
        return False
    if re.search(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', clean, re.IGNORECASE):
        return False
    if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', clean):
        return False

    # ✅ Starts with capital letter or digit
    if not re.match(r'^[A-Z0-9]', clean):
        return False

    return True



def extract_headings(doc) -> List[Dict[str, Any]]:
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

    for span in all_spans:
        if span['size'] < 8:
            continue  # Skip footnotes and footer-level text

        raw_text = span['text'].strip()
        clean = re.sub(r'[:\-–—\s]+$', '', raw_text)
        clean = re.sub(r'\s+', ' ', clean)

        if not is_heading_candidate(clean):
            continue

        # Numeric pattern logic for level detection
        level = None
        if re.match(r'^\d+\.\d+\.\d+', clean):  # 1.2.3
            level = "H3"
        elif re.match(r'^\d+\.\d+', clean):  # 1.2
            level = "H2"
        elif re.match(r'^\d+', clean):  # 1
            level = "H1"

        # Fall back to font size-based level if regex fails
        if not level:
        # fallback only if it's not common junk
            if len(clean.split()) >= 2 and len(clean) <= 60 and (span['flags'] & 2):
                level = size_to_level.get(span['size'])
            else:
                continue



        # For H1 and H2, prefer bold text
        if level in ('H1', 'H2') and not (span['flags'] & 2):
            continue

        headings.append({
            'level': level,
            'text': clean,
            'page': span['page'] + 1,
            'y': span['bbox'][1]
        })

    # Remove duplicates
    seen = set()
    unique_headings = []
    for h in headings:
        key = (h['level'], h['text'])  # remove page from uniqueness
        if key not in seen:
            unique_headings.append(h)
            seen.add(key)

    unique_headings.sort(key=lambda h: (h['page'], h.get('y', 0)))
    # Remove 'y' key before returning (optional for clean output)
    for h in unique_headings:
        h.pop('y', None)
    
    return unique_headings



def process_pdf(pdf_path: Path) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    title = extract_title(doc[0]) if len(doc) > 0 else ""
    outline = extract_headings(doc)
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