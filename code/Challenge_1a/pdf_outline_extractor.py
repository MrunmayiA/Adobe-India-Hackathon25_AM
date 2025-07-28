import os
# Suppress joblib CPU core warning by setting LOKY_MAX_CPU_COUNT
import multiprocessing
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.context")

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
try:
    import fitz  # try importing fitz directly
except ImportError:
    try:
        from PyMuPDF import fitz  # try importing from PyMuPDF
    except ImportError:
        print("Error: Could not import PyMuPDF. Please ensure it's installed correctly.")
        print("Try: pip install --upgrade --force-reinstall PyMuPDF==1.22.5")
        import sys
        sys.exit(1)

import numpy as np
import argparse
import re
from collections import defaultdict
from bs4 import BeautifulSoup
import html
import cssutils
import logging
cssutils.log.setLevel(logging.CRITICAL)  # Suppress cssutils warnings

# Constants for document processing
class DocumentConstants:
    # Heading levels
    H1: str = "H1"
    H2: str = "H2"
    H3: str = "H3"
    H4: str = "H4"
    BODY: str = "body"
    
    # File extensions
    PDF_EXT: str = ".pdf"
    HTML_EXT: str = ".html"
    JSON_EXT: str = ".json"
    
    # Font properties
    MIN_FONT_SIZE: float = 8.0
    MIN_HEADING_SIZE: float = 10.0
    MIN_TITLE_SIZE: float = 14.0
    FONT_SIZE_TOLERANCE: float = 0.5
    
    # Page layout
    PAGE_TOP_MARGIN_PCT: float = 0.35
    MIN_HEADING_SPACING: float = 30.0
    
    # Text properties
    MIN_TEXT_LENGTH: int = 3
    MAX_HEADING_LENGTH: int = 150
    
    # Font flags
    BOLD_FLAG: int = 2
    
    # Clustering
    NUM_FONT_CLUSTERS: int = 4

class OutputConstants:
    # JSON field names
    TITLE_FIELD: str = "title"
    OUTLINE_FIELD: str = "outline"
    LEVEL_FIELD: str = "level"
    TEXT_FIELD: str = "text"
    PAGE_FIELD: str = "page"
    
    # Output formatting
    INDENT_SPACES: int = 4
    ENSURE_ASCII: bool = False

class HeadingConstants:
    # Heading levels
    H1: str = "H1"
    H2: str = "H2"
    H3: str = "H3"
    
    # Common H1 sections
    MAJOR_SECTIONS: Set[str] = {
        "Revision History",
        "Table of Contents",
        "Acknowledgements",
        "Introduction",
        "References"
    }
    
    # Section numbering patterns
    H1_PATTERNS: List[str] = [
        r'^\d+\.\s+[A-Z][a-z]+',  # "1. Introduction"
        r'^\d+\.\s+[A-Z][a-zA-Z\s]+[a-zA-Z]$',  # "1. Foundation Level Extensions"
        r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$'  # "Foundation Level Extensions"
    ]
    
    H2_PATTERNS: List[str] = [
        r'^\d+\.\d+\s+[A-Z][a-z]+',  # "1.1 Overview"
        r'^\d+\.\d+\s+[A-Za-z\s]+'   # "2.1 Intended Audience"
    ]
    
    # Invalid patterns
    BULLET_PATTERN: str = r'^\d+\.\s+[A-Za-z].*[,\.]$'
    
    # Text validation
    MIN_LENGTH: int = 3
    MAX_LENGTH: int = 100
    
    # Font properties
    MIN_H1_SIZE: float = 14.0
    MIN_H2_SIZE: float = 12.0
    SIZE_TOLERANCE: float = 0.5
    
    # Filtering
    EXCLUDED_WORDS: Set[str] = {
        "overview",
        "version",
        "syllabus",
        "draft",
        "contents"
    }

class HeadingPatterns:
    # Common section headings
    COMMON_SECTIONS: Set[str] = {
        "Revision History",
        "Table of Contents",
        "Acknowledgements",
        "Introduction",
        "Executive Summary",
        "Conclusion"
    }
    
    # Form field patterns to ignore
    FORM_FIELDS: Set[str] = {
        "date",
        "name",
        "address",
        "signature"
    }
    
    # Regex patterns for section numbering
    SECTION_NUMBER: str = r'^(\d+\.)+\s'
    H1_SECTION: str = r'^[0-9]+\.\s+[A-Z]'
    H2_SECTION: str = r'^[0-9]+\.[0-9]+\s+'
    H3_SECTION: str = r'^[0-9]+\.[0-9]+\.[0-9]+\s+'
    BULLET_POINT: str = r'^[•\-]\s+'
    ALPHA_POINT: str = r'^[a-z]\)\s+'

class HTMLConstants:
    # HTML tags
    HEADING_TAGS: Set[str] = {'h1', 'h2', 'h3', 'h4'}
    CONTENT_TAGS: Set[str] = {'div', 'p', 'span'}
    ALL_CONTENT_TAGS: Set[str] = HEADING_TAGS | CONTENT_TAGS
    
    # CSS properties
    FONT_SIZE_PROP: str = 'font-size'
    FONT_WEIGHT_PROP: str = 'font-weight'
    TEXT_TRANSFORM_PROP: str = 'text-transform'
    MARGIN_TOP_PROP: str = 'margin-top'
    PAGE_BREAK_PROP: str = 'page-break-before'
    
    # CSS values
    UPPERCASE_VALUE: str = 'uppercase'
    ALWAYS_VALUE: str = 'always'
    
    # Thresholds
    MIN_FONT_SIZE: float = 14.0
    MIN_FONT_WEIGHT: float = 600.0
    MIN_MARGIN_TOP: float = 20.0

# --- Modular Extraction Functions ---

def extract_title(page) -> str:
    """Extract the title from the first page using largest, topmost text spans."""
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
    # Get all spans with max size in top portion of page
    title_spans = [span['text'] for span in spans 
                  if abs(span['size'] - max_size) < DocumentConstants.FONT_SIZE_TOLERANCE 
                  and span['bbox'][1] < page.rect.height * DocumentConstants.PAGE_TOP_MARGIN_PCT]
    if not title_spans:
        # fallback: just largest text
        title_spans = [span['text'] for span in spans 
                      if abs(span['size'] - max_size) < DocumentConstants.FONT_SIZE_TOLERANCE]
    title = ' '.join(title_spans)
    return title

def extract_title_from_html(soup: BeautifulSoup) -> str:
    """Extract title from HTML using various methods."""
    # Try <title> tag first
    title_tag = soup.find('title')
    if title_tag and title_tag.text.strip() and not title_tag.text.strip().endswith('.html'):
        return title_tag.text.strip()
    
    # Look for first large bold text in body
    body = soup.find('body')
    if not body:
        return ""
    
    # Look for elements with large font size or prominent styling
    for elem in body.find_all(['div', 'p', 'span']):
        style = elem.get('style', '')
        if style:
            # Parse CSS style
            style_dict = {prop.name: prop.value for prop in cssutils.parseStyle(style)}
            font_size = style_dict.get('font-size', '')
            font_weight = style_dict.get('font-weight', '')
            
            # Check if it's large and bold
            if (font_size and float(font_size.replace('px', '')) > 16) or \
               (font_weight and float(font_weight) >= 600):
                text = elem.get_text().strip()
                if text and len(text) > 3:  # Avoid short strings
                    return text
    
    return ""

# --- Outline Extraction ---
def extract_outline_from_bookmarks(doc) -> List[Dict[str, Any]]:
    toc = doc.get_toc(simple=False)
    outline = []
    for item in toc:
        level, text, page = item[0], item[1], item[2]
        if level == 1:
            lvl = DocumentConstants.H1
        elif level == 2:
            lvl = DocumentConstants.H2
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
    if not text or len(text.strip()) < DocumentConstants.MIN_TEXT_LENGTH or len(text.strip()) > DocumentConstants.MAX_HEADING_LENGTH:
        return False
    # Section numbering
    if re.match(HeadingPatterns.SECTION_NUMBER, text):
        return True
    # Major headings by name
    if text.strip() in HeadingPatterns.COMMON_SECTIONS:
        return True
    # Font size: must be larger than previous or body text
    if span['size'] >= prev_size and span['size'] > DocumentConstants.MIN_HEADING_SIZE:
        # Bold or all caps or centered
        if (span['flags'] & DocumentConstants.BOLD_FLAG) or text.isupper() or abs(span['bbox'][0] - 0) < DocumentConstants.MIN_HEADING_SPACING:
            return True
    # Position: more whitespace above than below (visual separation)
    if prev_y is not None and (span['bbox'][1] - prev_y) > DocumentConstants.MIN_HEADING_SPACING:
        return True
    return False

def is_valid_heading(text: str) -> bool:
    """Check if text is a valid heading."""
    clean_text = text.strip().lower()
    
    # Skip if too short or too long
    if len(clean_text) < HeadingConstants.MIN_LENGTH or len(clean_text) > HeadingConstants.MAX_LENGTH:
        return False
        
    # Skip if it's just an excluded word
    if clean_text in HeadingConstants.EXCLUDED_WORDS:
        return False
        
    # Skip if it ends with a comma or starts with a lowercase
    if clean_text.endswith(',') or (clean_text[0].isalpha() and clean_text[0].islower()):
        return False
        
    # Skip bullet points and incomplete sentences
    if re.match(HeadingConstants.BULLET_PATTERN, text.strip()):
        return False
        
    return True

def assign_heading_level(text: str, size: float, prev_level: str = None) -> str:
    """Assign heading level based on text pattern and font size."""
    # Skip invalid headings
    if not is_valid_heading(text):
        return None
        
    clean_text = text.strip()
        
    # Major sections are always H1
    if clean_text in HeadingConstants.MAJOR_SECTIONS:
        return HeadingConstants.H1
        
    # Check H1 patterns
    for pattern in HeadingConstants.H1_PATTERNS:
        if re.match(pattern, clean_text):
            return HeadingConstants.H1
            
    # Check H2 patterns
    for pattern in HeadingConstants.H2_PATTERNS:
        if re.match(pattern, clean_text):
            return HeadingConstants.H2
            
    # Use font size as fallback only for clear headings
    if not clean_text.endswith(('.', ',')) and not re.match(r'^\d+\.?\s+', clean_text):
        if size >= HeadingConstants.MIN_H1_SIZE:
            return HeadingConstants.H1
        elif size >= HeadingConstants.MIN_H2_SIZE:
            return HeadingConstants.H2
        
    return None

def cluster_font_sizes(spans: List[Dict[str, Any]], n_clusters=DocumentConstants.NUM_FONT_CLUSTERS) -> Dict[float, str]:
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
    """Extract outline using hybrid approach of patterns and font sizes."""
    all_spans = []
    prev_level = None
    outline = []
    
    # First pass: collect all text spans
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span['text'].strip()
                    if text:
                        all_spans.append({
                            'text': text,
                            'size': span['size'],
                            'page': page_num + 1,  # Convert to 1-based page numbers
                            'flags': span['flags'],
                            'bbox': span['bbox']
                        })
    
    # Second pass: identify headings
    seen_text = set()
    for span in all_spans:
        # Skip duplicate text
        if span['text'].strip() in seen_text:
            continue
            
        level = assign_heading_level(span['text'], span['size'], prev_level)
        if level:
            heading = {
                OutputConstants.LEVEL_FIELD: level,
                OutputConstants.TEXT_FIELD: span['text'].strip() + " ",
                OutputConstants.PAGE_FIELD: span['page']
            }
            outline.append(heading)
            prev_level = level
            seen_text.add(span['text'].strip())
    
    # Sort by page number
    outline.sort(key=lambda x: x[OutputConstants.PAGE_FIELD])
    
    return outline

def is_heading_candidate_html(elem, style_dict: dict) -> Tuple[bool, str]:
    """Determine if an element is a heading candidate and its level."""
    text = elem.get_text().strip()
    if not text or len(text) < DocumentConstants.MIN_TEXT_LENGTH:
        return False, ""
        
    # Check native heading tags
    if elem.name in HTMLConstants.HEADING_TAGS:
        return True, elem.name.upper()
    
    # Check styling
    font_size = style_dict.get(HTMLConstants.FONT_SIZE_PROP, '').replace('px', '')
    font_weight = style_dict.get(HTMLConstants.FONT_WEIGHT_PROP, '')
    text_transform = style_dict.get(HTMLConstants.TEXT_TRANSFORM_PROP, '')
    margin_top = style_dict.get(HTMLConstants.MARGIN_TOP_PROP, '').replace('px', '')
    
    try:
        is_large = font_size and float(font_size) > HTMLConstants.MIN_FONT_SIZE
        is_bold = font_weight and float(font_weight) >= HTMLConstants.MIN_FONT_WEIGHT
        is_caps = text_transform == HTMLConstants.UPPERCASE_VALUE or text.isupper()
        has_margin = margin_top and float(margin_top) > HTMLConstants.MIN_MARGIN_TOP
    except ValueError:
        return False, ""
    
    # H1 detection
    if (is_large and is_bold and (is_caps or has_margin)) or \
       re.match(HeadingPatterns.H1_SECTION, text):
        return True, DocumentConstants.H1
    
    # H2 detection
    if (is_bold and not is_caps) or \
       re.match(HeadingPatterns.H2_SECTION, text):
        return True, DocumentConstants.H2
    
    # H3 detection
    if re.match(HeadingPatterns.BULLET_POINT, text) or \
       re.match(HeadingPatterns.ALPHA_POINT, text) or \
       re.match(HeadingPatterns.H3_SECTION, text):
        return True, DocumentConstants.H3
    
    return False, ""

def get_page_number_html(elem) -> int:
    """Determine page number for an element."""
    # Look for page break indicators
    parent = elem.find_parent(style=re.compile(f"{HTMLConstants.PAGE_BREAK_PROP}:\\s*{HTMLConstants.ALWAYS_VALUE}"))
    if parent:
        # Count previous page breaks
        return len(parent.find_previous_siblings(style=re.compile(f"{HTMLConstants.PAGE_BREAK_PROP}:\\s*{HTMLConstants.ALWAYS_VALUE}")))
    return 0

def extract_outline_from_html(html_content: str) -> List[Dict[str, Any]]:
    """Extract outline from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    outline = []
    
    # Process all elements
    for elem in soup.find_all(list(HTMLConstants.ALL_CONTENT_TAGS)):
        # Skip empty elements
        if not elem.get_text().strip():
            continue
            
        # Get element styling
        style = elem.get('style', '')
        style_dict = {prop.name: prop.value for prop in cssutils.parseStyle(style)}
        
        # Check if it's a heading
        is_heading, level = is_heading_candidate_html(elem, style_dict)
        if is_heading:
            text = elem.get_text().strip()
            
            # Skip form fields and boilerplate
            if text.lower() in HeadingPatterns.FORM_FIELDS:
                continue
                
            # Skip if it looks like a header/footer
            if elem.find_previous_siblings(string=text) or \
               elem.find_next_siblings(string=text):
                continue
            
            # Add to outline
            outline.append({
                'level': level,
                'text': text if text.endswith(' ') else text + ' ',
                'page': get_page_number_html(elem)
            })
    
    return outline

def analyze_text_properties(doc) -> Tuple[Dict, List]:
    """Analyze text properties across the document to understand heading structure."""
    # Store text properties for analysis
    font_sizes = defaultdict(list)  # size -> [(text, page, y_pos)]
    font_styles = defaultdict(list)  # (size, flags) -> [(text, page)]
    section_numbers = []  # [(text, page, size, flags)]
    major_sections = []   # [(text, page, size, flags)]
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")['blocks']
        for block in blocks:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    text = span['text'].strip()
                    if not text:
                        continue
                    
                    # Store by font size
                    font_sizes[span['size']].append((text, page_num + 1, span['bbox'][1]))
                    
                    # Store by font style (size + bold/italic)
                    style_key = (span['size'], span['flags'])
                    font_styles[style_key].append((text, page_num + 1))
                    
                    # Check for section numbering
                    if re.match(HeadingPatterns.SECTION_NUMBER, text):
                        section_numbers.append((text, page_num + 1, span['size'], span['flags']))
                    
                    # Check for major section names
                    if text in HeadingPatterns.COMMON_SECTIONS:
                        major_sections.append((text, page_num + 1, span['size'], span['flags']))
    
    # Sort font sizes
    sorted_sizes = sorted(font_sizes.keys(), reverse=True)
    size_analysis = {}
    for i, size in enumerate(sorted_sizes[:5]):  # Look at top 5 sizes
        texts = font_sizes[size]
        size_analysis[size] = {
            'count': len(texts),
            'examples': texts[:3],  # First 3 examples
            'likely_level': f"H{i+1}" if i < 3 else DocumentConstants.BODY
        }
    
    return size_analysis, section_numbers

def print_heading_analysis(doc_path: str):
    """Print detailed analysis of heading structure in the PDF."""
    doc = fitz.open(doc_path)
    print(f"\nAnalyzing PDF: {doc_path}")
    print("=" * 80)
    
    # Get built-in TOC if available
    toc = doc.get_toc()
    if toc:
        print("\nBuilt-in TOC/Bookmarks found:")
        print("-" * 40)
        for level, title, page in toc:
            print(f"{'  ' * (level-1)}Level {level}: {title} (page {page})")
    else:
        print("\nNo built-in TOC/Bookmarks found.")
    
    # Analyze text properties
    size_analysis, section_numbers = analyze_text_properties(doc)
    
    print("\nFont Size Analysis:")
    print("-" * 40)
    for size, info in size_analysis.items():
        print(f"\nSize {size:.1f} ({info['likely_level']}) - {info['count']} occurrences")
        print("Examples:")
        for text, page, y_pos in info['examples']:
            print(f"  - '{text}' (page {page})")
    
    if section_numbers:
        print("\nSection Numbering Patterns:")
        print("-" * 40)
        for text, page, size, flags in section_numbers:
            depth = text.count('.') + 1
            level = 'H' + str(depth)
            bold = "bold" if flags & 2 else "normal"
            print(f"Level {level}: '{text}' (page {page}, size {size:.1f}, {bold})")
    
    # Close the document
    doc.close()
    print("\n" + "=" * 80)

def analyze_pdf(pdf_path: str) -> Dict[str, Any]:
    """Analyze a PDF and return structured heading information."""
    doc = fitz.open(pdf_path)
    size_analysis, section_numbers = analyze_text_properties(doc)
    
    # Group headings by level
    headings_by_level = defaultdict(list)
    
    # First, process section numbers
    for text, page, size, flags in section_numbers:
        depth = text.count('.') + 1
        level = f"H{depth}"
        headings_by_level[level].append({
            'text': text,
            'page': page,
            'size': size,
            'bold': bool(flags & 2)
        })
    
    # Then, look at font sizes for non-numbered headings
    for size, info in size_analysis.items():
        level = info['likely_level']
        if level.startswith('H'):
            for text, page, _ in info['examples']:
                if not any(text in h['text'] for h in headings_by_level[level]):
                    headings_by_level[level].append({
                        'text': text,
                        'page': page,
                        'size': size
                    })
    
    return dict(headings_by_level)

# --- Main PDF Processing ---
def process_pdf(pdf_path: Path) -> dict:
    """Process PDF and return structured outline."""
    doc = fitz.open(pdf_path)
    
    # Extract title from first page
    title = extract_title(doc[0]) if len(doc) > 0 else ""
    title = title.strip() + "  "  # Add consistent spacing
    
    # Extract outline
    outline = extract_outline_hybrid(doc)
    
    # Format output
    return {
        OutputConstants.TITLE_FIELD: title,
        OutputConstants.OUTLINE_FIELD: outline
    }

def process_html(html_path: Path) -> dict:
    """Process an HTML file and return structured outline."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    title = extract_title_from_html(soup)
    outline = extract_outline_from_html(html_content)
    
    # Form detection similar to PDF version
    if (len(outline) == 1 and outline[0]['text'].strip() == title.strip()):
        outline = []
        
    return {
        "title": title,
        "outline": outline
    }

def process_file(file_path: Path) -> dict:
    """Process either PDF or HTML file."""
    if file_path.suffix.lower() == DocumentConstants.PDF_EXT:
        return process_pdf(file_path)
    elif file_path.suffix.lower() == DocumentConstants.HTML_EXT:
        return process_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

def process_all_files(input_dir: Path, output_dir: Path):
    """Process all PDF files in input directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = list(input_dir.glob(f"*{DocumentConstants.PDF_EXT}"))
    
    for input_file in input_files:
        try:
            result = process_pdf(input_file)
            output_file = output_dir / f"{input_file.stem}{DocumentConstants.JSON_EXT}"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, 
                         indent=OutputConstants.INDENT_SPACES,
                         ensure_ascii=OutputConstants.ENSURE_ASCII)
            print(f"Processed {input_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Failed to process {input_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="PDF/HTML Outline Extractor")
    parser.add_argument('--input', type=str, default=os.environ.get("INPUT_DIR", "input"), 
                      help='Input directory for PDFs and HTMLs')
    parser.add_argument('--output', type=str, default=os.environ.get("OUTPUT_DIR", "output"), 
                      help='Output directory for JSONs')
    parser.add_argument('--analyze', action='store_true', help='Print detailed heading analysis')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if args.analyze:
        # Try to analyze first PDF or HTML file
        pdf_files = list(input_dir.glob(f"*{DocumentConstants.PDF_EXT}"))
        html_files = list(input_dir.glob(f"*{DocumentConstants.HTML_EXT}"))
        if pdf_files:
            print_heading_analysis(str(pdf_files[0]))
            analysis = analyze_pdf(str(pdf_files[0]))
            print("\nStructured Heading Analysis:")
            for level, headings in analysis.items():
                print(f"\n{level}:")
                for h in headings:
                    print(f"  - {h['text']} (page {h['page']}, size {h['size']:.1f})")
        elif html_files:
            print(f"\nAnalyzing HTML: {html_files[0]}")
            with open(html_files[0], 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            print("\nTitle:", extract_title_from_html(soup))
            outline = extract_outline_from_html(f.read())
            print("\nOutline:")
            for item in outline:
                print(f"{item['level']}: {item['text']} (page {item['page']})")
    else:
        process_all_files(input_dir, output_dir)

if __name__ == "__main__":
    main() 