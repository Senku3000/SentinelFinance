"""Multi-format document parser for financial documents.

Supports: PDF, TXT, MD, Excel (.xlsx/.xls), CSV, and images (via OCR).
Delegates PDF/TXT/MD to existing PDFParser. Adds tabular and image support.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from .pdf_parser import PDFParser, DocumentChunk


class DocumentParser:
    """Parses financial documents in multiple formats into text chunks."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".md",          # existing
        ".xlsx", ".xls", ".csv",        # tabular
        ".png", ".jpg", ".jpeg",        # images (OCR)
    }

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._pdf_parser = PDFParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def parse_file(self, file_path: Path) -> List[DocumentChunk]:
        """Parse any supported file into chunks."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext in (".pdf", ".txt", ".md"):
            return self._pdf_parser.parse_file(file_path)
        elif ext in (".xlsx", ".xls"):
            return self._parse_excel(file_path)
        elif ext == ".csv":
            return self._parse_csv(file_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            return self._parse_image(file_path)
        else:
            print(f"Unsupported file type: {ext}")
            return []

    def _parse_excel(self, file_path: Path) -> List[DocumentChunk]:
        """Parse Excel file — convert each sheet's rows to text lines, then chunk."""
        try:
            xls = pd.ExcelFile(file_path, engine="openpyxl")
        except Exception as e:
            print(f"Error reading Excel file {file_path}: {e}")
            return []

        all_text_parts = []
        for sheet_name in xls.sheet_names:
            try:
                df = xls.parse(sheet_name)
                df = df.dropna(how="all")
                if df.empty:
                    continue

                sheet_text = self._dataframe_to_text(df, sheet_name)
                all_text_parts.append(sheet_text)
            except Exception as e:
                print(f"Error parsing sheet '{sheet_name}' in {file_path}: {e}")

        if not all_text_parts:
            return []

        full_text = "\n\n".join(all_text_parts)
        return self._pdf_parser._split_text(full_text, file_path)

    def _parse_csv(self, file_path: Path) -> List[DocumentChunk]:
        """Parse CSV file — convert rows to text lines, then chunk."""
        try:
            df = pd.read_csv(file_path)
            df = df.dropna(how="all")
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return []

        if df.empty:
            return []

        full_text = self._dataframe_to_text(df)
        return self._pdf_parser._split_text(full_text, file_path)

    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: Optional[str] = None) -> str:
        """Convert a DataFrame to structured text lines.

        Each row becomes: "Column1: Value1 | Column2: Value2 | ..."
        This preserves column context per row for embedding and LLM extraction.
        """
        columns = [str(c).strip() for c in df.columns]
        lines = []

        if sheet_name:
            lines.append(f"--- Sheet: {sheet_name} ---")

        # Add header summary
        lines.append(f"Columns: {', '.join(columns)}")
        lines.append("")

        for _, row in df.iterrows():
            parts = []
            for col in columns:
                val = row.get(col, "")
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
            if parts:
                lines.append(" | ".join(parts))

        return "\n".join(lines)

    def _parse_image(self, file_path: Path) -> List[DocumentChunk]:
        """Parse image via OCR (requires pytesseract + Pillow)."""
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            print(
                "Image OCR requires 'pytesseract' and 'Pillow'. "
                "Install them with: pip install pytesseract Pillow\n"
                "You also need Tesseract OCR installed on your system."
            )
            return []

        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
        except Exception as e:
            print(f"Error running OCR on {file_path}: {e}")
            return []

        if not text.strip():
            print(f"No text extracted from image {file_path}")
            return []

        return self._pdf_parser._split_text(text, file_path)

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if a file extension is supported."""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS
