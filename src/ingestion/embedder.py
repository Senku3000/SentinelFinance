"""Document embedding and vector database population"""

from pathlib import Path
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle

from ..config import Config
from .pdf_parser import PDFParser, DocumentChunk


class DocumentEmbedder:
    """Handles document embedding and vector database population"""
    
    def __init__(self):
        """Initialize embedder"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.parser = PDFParser()
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize FAISS vectorstore"""
        faiss_path = Path(Config.CHROMA_DB_PATH) / "faiss_index"
        
        try:
            # Try to load existing index
            if faiss_path.exists():
                self.vectorstore = FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {faiss_path}")
            else:
                # Create new empty index
                # FAISS needs at least one document to initialize, so we'll create it on first add
                self.vectorstore = None
                print("Creating new FAISS index (will be saved on first document)")
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")
            self.vectorstore = None
    
    def _save_vectorstore(self):
        """Save FAISS index to disk"""
        if self.vectorstore:
            faiss_path = Path(Config.CHROMA_DB_PATH) / "faiss_index"
            faiss_path.mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(str(faiss_path))
    
    def ingest_documents(self, documents_path: Path = None) -> int:
        """
        Ingest all documents from the documents directory
        
        Args:
            documents_path: Path to documents directory (defaults to Config.DOCUMENTS_PATH)
            
        Returns:
            Number of documents ingested
        """
        if documents_path is None:
            documents_path = Config.get_documents_path()
        
        if not documents_path.exists():
            print(f"Documents directory does not exist: {documents_path}")
            return 0
        
        # Find all PDF and text files
        pdf_files = list(documents_path.glob("*.pdf"))
        text_files = list(documents_path.glob("*.txt")) + list(documents_path.glob("*.md"))
        
        all_files = pdf_files + text_files
        
        if not all_files:
            print(f"No documents found in {documents_path}")
            return 0
        
        print(f"Found {len(all_files)} documents to ingest")
        
        total_chunks = 0
        all_documents = []
        
        for file_path in all_files:
            print(f"Processing: {file_path.name}")
            
            # Parse file into chunks
            chunks = self.parser.parse_file(file_path)
            
            if not chunks:
                print(f"  No chunks extracted from {file_path.name}")
                continue
            
            # Convert to LangChain Documents
            documents = [
                Document(
                    page_content=chunk.content,
                    metadata=chunk.metadata
                )
                for chunk in chunks
            ]
            
            all_documents.extend(documents)
            total_chunks += len(chunks)
            print(f"  Prepared {len(chunks)} chunks from {file_path.name}")
        
        # Add all documents to vectorstore at once
        if all_documents:
            try:
                if self.vectorstore is None:
                    # Create new FAISS index
                    self.vectorstore = FAISS.from_documents(all_documents, self.embeddings)
                    print("Created new FAISS index")
                else:
                    # Add to existing index
                    self.vectorstore.add_documents(all_documents)
                    print("Added documents to existing FAISS index")
                
                # Save to disk
                self._save_vectorstore()
                print(f"\nTotal chunks ingested: {total_chunks}")
            except Exception as e:
                print(f"Error adding documents: {e}")
                return 0
        
        return total_chunks
    
    def ingest_file(self, file_path: Path) -> int:
        """
        Ingest a single file
        
        Args:
            file_path: Path to file to ingest
            
        Returns:
            Number of chunks ingested
        """
        if not file_path.exists():
            print(f"File does not exist: {file_path}")
            return 0
        
        print(f"Processing: {file_path.name}")
        
        # Parse file
        chunks = self.parser.parse_file(file_path)
        
        if not chunks:
            return 0
        
        # Convert to LangChain Documents
        documents = [
            Document(
                page_content=chunk.content,
                metadata=chunk.metadata
            )
            for chunk in chunks
        ]
        
        # Add to vectorstore
        try:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vectorstore.add_documents(documents)
            
            self._save_vectorstore()
            print(f"Added {len(chunks)} chunks from {file_path.name}")
            return len(chunks)
        except Exception as e:
            print(f"Error adding chunks: {e}")
            return 0
