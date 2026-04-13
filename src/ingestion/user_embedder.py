"""Per-user document embedding and FAISS index management.

Each user gets their own FAISS index at data/user_profiles/{user_id}/faiss_index/,
separate from the global knowledge base index.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..config import Config
from .document_parser import DocumentParser


class UserEmbedder:
    """Manages per-user FAISS indexes for personal financial documents."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.parser = DocumentParser()

    def _load_user_vectorstore(self, user_id: str) -> Optional[FAISS]:
        """Load a user's FAISS index from disk, or None if it doesn't exist."""
        faiss_path = Config.get_user_faiss_path(user_id)
        index_file = faiss_path / "index.faiss"

        if not index_file.exists():
            return None

        try:
            return FAISS.load_local(
                str(faiss_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Warning: Could not load user FAISS index for {user_id}: {e}")
            return None

    def _save_user_vectorstore(self, user_id: str, vectorstore: FAISS):
        """Save a user's FAISS index to disk."""
        faiss_path = Config.get_user_faiss_path(user_id)
        vectorstore.save_local(str(faiss_path))

    def _load_manifest(self, user_id: str) -> List[Dict[str, Any]]:
        """Load the document manifest for a user."""
        manifest_file = Config.get_user_manifest_file(user_id)
        if not manifest_file.exists():
            return []
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_manifest(self, user_id: str, manifest: List[Dict[str, Any]]):
        """Save the document manifest for a user."""
        manifest_file = Config.get_user_manifest_file(user_id)
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    def ingest_user_document(self, user_id: str, file_path: Path) -> int:
        """Parse, embed, and store a document in the user's personal FAISS index.

        Args:
            user_id: User identifier
            file_path: Path to the document file

        Returns:
            Number of chunks ingested
        """
        file_path = Path(file_path)

        # Copy file to user's documents directory
        docs_dir = Config.get_user_documents_path(user_id)
        dest_path = docs_dir / file_path.name
        if file_path != dest_path:
            shutil.copy2(file_path, dest_path)

        # Parse into chunks
        chunks = self.parser.parse_file(dest_path)
        if not chunks:
            print(f"No chunks extracted from {file_path.name}")
            return 0

        # Convert to LangChain Documents with user metadata
        documents = [
            Document(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "user_id": user_id,
                    "document_type": "personal",
                }
            )
            for chunk in chunks
        ]

        # Add to user's FAISS index
        vectorstore = self._load_user_vectorstore(user_id)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            vectorstore.add_documents(documents)

        self._save_user_vectorstore(user_id, vectorstore)

        # Update manifest
        manifest = self._load_manifest(user_id)

        # Remove old entry for same filename if re-uploading
        manifest = [m for m in manifest if m["filename"] != file_path.name]

        manifest.append({
            "filename": file_path.name,
            "file_type": file_path.suffix.lower(),
            "upload_date": datetime.now().isoformat(),
            "num_chunks": len(chunks),
            "extraction_status": "pending",
        })
        self._save_manifest(user_id, manifest)

        print(f"Ingested {len(chunks)} chunks from {file_path.name} for user {user_id}")
        return len(chunks)

    def search_user_documents(
        self, user_id: str, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search a user's personal document index.

        Args:
            user_id: User identifier
            query: Search query
            k: Number of results to return

        Returns:
            List of results with content, metadata, and relevance scores
        """
        vectorstore = self._load_user_vectorstore(user_id)
        if vectorstore is None:
            return []

        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                    "retrieval_type": "personal_document",
                }
                for doc, score in results
            ]
        except Exception as e:
            print(f"Error searching user documents for {user_id}: {e}")
            return []

    def delete_user_document(self, user_id: str, filename: str) -> bool:
        """Delete a document and rebuild the user's FAISS index.

        FAISS doesn't support individual vector deletion, so we rebuild
        from all remaining documents.

        Args:
            user_id: User identifier
            filename: Name of the file to delete

        Returns:
            True if successful
        """
        docs_dir = Config.get_user_documents_path(user_id)
        file_path = docs_dir / filename

        # Remove the file
        if file_path.exists():
            file_path.unlink()

        # Remove from manifest
        manifest = self._load_manifest(user_id)
        manifest = [m for m in manifest if m["filename"] != filename]
        self._save_manifest(user_id, manifest)

        # Delete old FAISS index
        faiss_path = Config.get_user_faiss_path(user_id)
        if faiss_path.exists():
            shutil.rmtree(faiss_path)
            faiss_path.mkdir(parents=True, exist_ok=True)

        # Rebuild from remaining documents
        remaining_files = list(docs_dir.iterdir())
        for f in remaining_files:
            if DocumentParser.is_supported(f):
                self.ingest_user_document(user_id, f)

        print(f"Deleted {filename} and rebuilt index for user {user_id}")
        return True

    def list_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """List all uploaded documents for a user.

        Returns:
            List of document metadata from the manifest
        """
        return self._load_manifest(user_id)

    def has_documents(self, user_id: str) -> bool:
        """Check if a user has any uploaded documents."""
        return len(self._load_manifest(user_id)) > 0
