"""Script to ingest documents into vector database"""

from pathlib import Path
from src.ingestion.embedder import DocumentEmbedder
from src.config import Config

if __name__ == "__main__":
    print("Starting document ingestion...")
    print(f"Documents path: {Config.DOCUMENTS_PATH}")
    print(f"Vector DB path: {Config.VECTOR_DB_PATH}")
    print()
    
    embedder = DocumentEmbedder()
    
    # Ingest all documents
    num_chunks = embedder.ingest_documents()
    
    print(f"\nIngestion complete! Total chunks: {num_chunks}")
