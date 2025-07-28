import os
import json
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import numpy as np
from datetime import datetime
import re
from sklearn.cluster import KMeans

class DocumentAnalyzer:
    def __init__(self, persona: str = "", job: str = ""):
        self.persona = persona
        self.job = job
        
    def _extract_title(self, page) -> str:
        """Enhanced title extraction with better multi-line handling"""
        try:
            blocks = page.get_text("dict")['blocks']
            spans = []
            for block in blocks:
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        if span['text'].strip():
                            spans.append(span)
            if not spans:
                return ""

            # Consider spans in the top 25% of the page with largest font sizes
            page_height = page.rect.height
            top_spans = [s for s in spans if s['bbox'][1] < page_height * 0.25]
            if not top_spans:
                return ""

            # Find the largest font size in the top area
            max_size = max(span['size'] for span in top_spans)
            title_spans = [s for s in top_spans if abs(s['size'] - max_size) < 1.0]
            title_spans.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))

            # Group nearby spans
            title_parts = []
            current_group = []
            prev_y = None

            for span in title_spans:
                if prev_y is None or abs(span['bbox'][1] - prev_y) < 40:
                    current_group.append(span['text'].strip())
                else:
                    if current_group:
                        title_parts.append(" ".join(current_group))
                    current_group = [span['text'].strip()]
                prev_y = span['bbox'][1]

            if current_group:
                title_parts.append(" ".join(current_group))

            return " ".join(title_parts).strip()
        except Exception as e:
            print(f"Warning: Error extracting title: {e}")
            return ""

    def _is_heading_candidate(self, text: str, flags: int = 0) -> bool:
        """Enhanced heading detection with better list item filtering"""
        clean = text.strip().rstrip(':').strip()
        
        if not clean or len(clean) < 3 or len(clean) > 150:
            return False

        # Skip obvious non-headings
        if any([
            len(clean.split()) > 20,  # Too many words
            re.match(r'^[a-z]', clean),  # Starts with lowercase
            re.match(r'Page\s+\d+\s+of\s+\d+', clean, re.IGNORECASE),  # Page numbers
            re.match(r'Version\s+\d+(\.\d+)*', clean, re.IGNORECASE),  # Version numbers
            re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', clean),  # Dates
            re.match(r'^[\u2022\u2023\u2043\u2219\u25E6\u2043\u2022]\s', clean),  # Bullet points
        ]):
            return False

        # Handle numbered patterns
        if re.match(r'^\d+', clean):
            # Accept if it's a section number (e.g., "1. Title" or "1.1 Title")
            if re.match(r'^\d+(\.\d+)*\s+[A-Z]', clean):
                return True
            # Reject if it looks like a list item
            if re.match(r'^\d+[\.\)]\s+[a-z]', clean) or len(clean.split()) > 8:
                return False

        # Must start with capital letter or section number
        if not re.match(r'^(\d+(\.\d+)*\s+)?[A-Z]', clean):
            return False

        return True

    def _get_heading_level(self, text: str, font_size: float, size_clusters: Dict[float, str]) -> str:
        """Determine heading level using both formatting and content patterns"""
        clean = text.strip()
        if re.match(r'^\d+\.\d+\.\d+', clean):  # X.Y.Z
            return "H3"
        elif re.match(r'^\d+\.\d+\s', clean):   # X.Y
            return "H2"
        elif re.match(r'^\d+\.\s', clean):      # X.
            return "H1"
        
        return size_clusters.get(font_size, "H3")

    def _extract_section_content(self, page, bbox) -> str:
        """Extract and clean the content of a section"""
        try:
            content = page.get_text("text", clip=bbox)
            return re.sub(r'\s+', ' ', content).strip()
        except Exception as e:
            print(f"Warning: Error extracting section content: {e}")
            return ""

    def analyze_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze a single PDF document and extract structured information"""
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            if doc.page_count == 0:
                print(f"Warning: {pdf_path} has no pages")
                return {"title": "", "sections": []}

            # Extract title from first page
            title = self._extract_title(doc[0])
            
            # Collect all text spans
            all_spans = []
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    blocks = page.get_text("dict")['blocks']
                    for block in blocks:
                        for line in block.get('lines', []):
                            for span in line.get('spans', []):
                                if span['text'].strip():
                                    all_spans.append({**span, 'page': page_num})
                except Exception as e:
                    print(f"Warning: Error processing page {page_num}: {e}")
                    continue

            if not all_spans:
                return {"title": title, "sections": []}

            # Cluster font sizes
            sizes = np.array([span['size'] for span in all_spans]).reshape(-1, 1)
            n_clusters = min(4, len(set(sizes.flatten())))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(sizes)
            centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
            
            size_to_level = {}
            for sz in set(sizes.flatten()):
                idx = np.argmin([abs(sz - c) for c in centers])
                size_to_level[sz] = f"H{idx + 1}"

            # Extract sections
            sections = []
            current_section = None
            prev_heading_level = None

            for span in all_spans:
                text = span['text'].strip()
                if not text:
                    continue

                if self._is_heading_candidate(text, span['flags']):
                    level = self._get_heading_level(text, span['size'], size_to_level)
                    
                    # Ensure proper heading hierarchy
                    if prev_heading_level:
                        level_num = int(level[1])
                        prev_num = int(prev_heading_level[1])
                        if level_num > prev_num + 1:
                            level = f"H{prev_num + 1}"
                    
                    if current_section:
                        try:
                            page = doc[current_section['page']]
                            content = self._extract_section_content(page, current_section['bbox'])
                            current_section['content'] = content
                        except Exception as e:
                            print(f"Warning: Could not extract content: {e}")
                            current_section['content'] = ""

                    current_section = {
                        'text': text,
                        'level': level,
                        'page': span['page'],
                        'bbox': span['bbox']
                    }
                    sections.append(current_section)
                    prev_heading_level = level

            # Process the last section
            if current_section:
                try:
                    page = doc[current_section['page']]
                    content = self._extract_section_content(page, current_section['bbox'])
                    current_section['content'] = content
                except Exception as e:
                    print(f"Warning: Could not extract content from last section: {e}")
                    current_section['content'] = ""

            # Post-process sections
            final_sections = []
            for section in sections:
                # Convert to 1-based page numbers for output
                section['page'] = section['page'] + 1
                # Calculate importance score
                section['importance_rank'] = self._calculate_importance(
                    section['text'],
                    section['level'],
                    section.get('content', ''),
                    section['page'],
                    len(sections)
                )
                # Remove internal fields
                section.pop('bbox', None)
                section.pop('content', None)
                final_sections.append(section)

            return {
                "title": title,
                "sections": final_sections
            }

        except Exception as e:
            import traceback
            print(f"Error analyzing {pdf_path}:")
            print(traceback.format_exc())
            return {"title": "", "sections": []}
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    print(f"Warning: Error closing document: {e}")

    def _calculate_importance(self, text: str, level: str, content: str, page_num: int, total_sections: int) -> float:
        """Calculate section importance score"""
        score = 0.0
        
        # Position-based importance (earlier sections often more important)
        position_score = 1 - ((page_num - 1) / total_sections)
        score += position_score * 0.3

        # Heading level importance
        level_scores = {"H1": 1.0, "H2": 0.7, "H3": 0.4}
        score += level_scores.get(level, 0.2) * 0.3

        # Content-based importance
        content_length = len(content.split())
        if content_length > 500:
            score += 0.2
        elif content_length > 200:
            score += 0.1

        # Relevance to persona/job
        text_lower = text.lower()
        important_patterns = [
            r'overview', r'introduction', r'summary', r'conclusion',
            r'result', r'finding', r'key', r'important', r'critical',
            r'main', r'essential', r'primary', r'fundamental'
        ]
        matches = sum(1 for pattern in important_patterns if re.search(pattern, text_lower))
        score += (matches * 0.1)

        return min(1.0, score)

    def process_documents(self, input_paths: List[Path]) -> Dict[str, Any]:
        """Process multiple documents and generate the final output"""
        results = []
        for path in input_paths:
            try:
                print(f"\nProcessing document: {path}")
                doc_result = self.analyze_document(path)
                results.append({
                    "file_name": path.name,
                    **doc_result
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")

        return {
            "metadata": {
                "input_documents": [str(p) for p in input_paths],
                "persona": self.persona,
                "job_to_be_done": self.job,
                "processing_timestamp": datetime.now().isoformat()
            },
            "documents": results
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced PDF Document Analyzer")
    parser.add_argument('--input', type=str, required=True, help='Input directory or file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--persona', type=str, default="", help='Persona description')
    parser.add_argument('--job', type=str, default="", help='Job to be done')
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if input_path.is_file():
        input_paths = [input_path]
    else:
        input_paths = list(input_path.glob("*.pdf"))
        if not input_paths:
            print(f"No PDF files found in {input_path}")
            return
        input_paths = [p.resolve() for p in input_paths]

    analyzer = DocumentAnalyzer(args.persona, args.job)
    results = analyzer.process_documents(input_paths)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults written to {output_path}")

if __name__ == "__main__":
    main()
