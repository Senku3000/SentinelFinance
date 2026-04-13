"""PDF and text document parser"""

from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
import pypdf


class DocumentChunk(BaseModel):
    """Represents a chunk of a document"""
    content: str
    metadata: Dict[str, Any]
    chunk_index: int


class PDFParser:
    """Parser for PDF and text documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize parser
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def parse_pdf(self, file_path: Path) -> List[DocumentChunk]:
        """
        Parse a PDF file into chunks
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                # Split into chunks
                chunks = self._split_text(full_text, file_path)
                
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            # Return empty chunks on error
        
        return chunks
    
    def parse_text(self, file_path: Path) -> List[DocumentChunk]:
        """
        Parse a text file into chunks
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = self._split_text(text, file_path)
        except Exception as e:
            print(f"Error parsing text file {file_path}: {e}")
        
        return chunks
    
    def _split_text(self, text: str, file_path: Path) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            file_path: Source file path
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # Simple chunking by character count
        # In production, you might want sentence-aware chunking
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "chunk_index": chunk_index
                    },
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def parse_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Parse a file (PDF or text) into chunks
        
        Args:
            file_path: Path to file
            
        Returns:
            List of document chunks
        """
        if file_path.suffix.lower() == '.pdf':
            return self.parse_pdf(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self.parse_text(file_path)
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return []
