"""User Document Tool - searches user's personal FAISS index."""

from typing import Dict, Any, Optional, List
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from ..ingestion.user_embedder import UserEmbedder


class UserDocumentTool(BaseTool):
    """Tool for searching user's personal uploaded financial documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "user_document_tool"
    description: str = """
    Search through the user's uploaded personal financial documents.
    Use this to find specific information from documents like salary slips,
    tax returns, expense sheets, bank statements, etc.
    """

    def _run(
        self,
        query: str,
        user_id: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Search user's personal document index.

        Args:
            query: Search query
            user_id: User identifier
            k: Number of results

        Returns:
            Dictionary with search results
        """
        try:
            embedder = UserEmbedder()

            if not embedder.has_documents(user_id):
                return {
                    "success": False,
                    "error": "No documents uploaded for this user",
                    "results": [],
                }

            results = embedder.search_user_documents(user_id, query, k=k)

            return {
                "success": True,
                "results": results,
                "num_results": len(results),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
            }

    async def _arun(
        self,
        query: str,
        user_id: str,
        k: int = 5,
    ) -> Dict[str, Any]:
        """Async version of _run."""
        return self._run(query, user_id, k)


def create_user_document_tool() -> UserDocumentTool:
    """Factory function to create UserDocumentTool instance."""
    return UserDocumentTool()
