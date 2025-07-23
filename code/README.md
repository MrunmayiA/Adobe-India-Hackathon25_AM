# PDF Outline Extractor

## Overview

This tool extracts a structured outline (title, H1, H2, H3, H4 headings) from PDF files and outputs the results as JSON, following the Adobe India Hackathon 2025 requirements.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your PDF files in the `input/` directory (create if it doesn't exist).

## Usage

Run the extractor:
```bash
python pdf_outline_extractor.py
```

- By default, it reads from `input/` and writes JSON outputs to `output/`.
- You can override directories with environment variables:
  - `PDF_INPUT_DIR` (default: `input`)
  - `PDF_OUTPUT_DIR` (default: `output`)

## Output Format

Each output JSON will look like:
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Heading 1", "page": 1 },
    { "level": "H2", "text": "Subheading", "page": 2 }
  ]
}
```

## Notes
- Uses PyMuPDF (fitz) and numpy for PDF parsing and clustering.
- Handles multiple PDFs in batch.
- Designed for offline, fast, and robust operation. 