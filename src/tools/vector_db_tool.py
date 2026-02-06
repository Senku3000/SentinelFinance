"""Vector DB Tool - RAG retrieval using FAISS"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path

from ..config import Config


class VectorDBTool(BaseTool):
    """Tool for retrieving financial knowledge from vector database"""
    
    name: str = "vector_db_tool"
    description: str = """
    Use this tool to search for financial knowledge, tax laws, investment principles,
    and regulations from the knowledge base.
    
    This tool searches through:
    - Indian Tax Code sections (80C, 80D, 24(b), etc.)
    - Investment strategies and principles
    - Financial planning frameworks
    - Market norms and benchmarks
    - Regulatory information
    
    Input should be a search query describing what financial information you need.
    """
    
    vectorstore: Optional[FAISS] = None
    embeddings: Optional[HuggingFaceEmbeddings] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize FAISS vectorstore"""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL
            )
            
            # Try to load existing FAISS index
            faiss_path = Path(Config.CHROMA_DB_PATH) / "faiss_index"
            
            if faiss_path.exists():
                try:
                    self.vectorstore = FAISS.load_local(
                        str(faiss_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print("Loaded FAISS vectorstore")
                except Exception as e:
                    print(f"Warning: Could not load FAISS index: {e}")
                    self.vectorstore = None
            else:
                print("FAISS index not found. Please run document ingestion first.")
                self.vectorstore = None
            
        except Exception as e:
            print(f"Warning: Could not initialize vectorstore: {e}")
            print("Vector DB will be available after document ingestion.")
            self.vectorstore = None
    
    def _run(
        self,
        query: str,
        k: int = 5,
        rewrite_query: bool = True
    ) -> Dict[str, Any]:
        """
        Search the vector database for relevant financial knowledge
        
        Args:
            query: Search query
            k: Number of results to return
            rewrite_query: Whether to rewrite query for better retrieval
            
        Returns:
            Dictionary with search results and relevance scores
        """
        if not self.vectorstore:
            return {
                "success": False,
                "error": "Vector database not initialized. Please run document ingestion first.",
                "results": [],
                "scores": []
            }
        
        try:
            # Query rewriting could be done here with LLM
            # For now, use query as-is
            search_query = query
            
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                search_query,
                k=k
            )
            
            # Format results
            formatted_results = []
            scores = []
            
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
                scores.append(float(score))
            
            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "scores": scores,
                "count": len(formatted_results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "scores": []
            }
    
    def _rewrite_query(self, original_query: str, user_context: Optional[Dict] = None) -> str:
        """
        Rewrite query for better retrieval
        This could use an LLM to expand/refine the query
        """
        # Simple implementation - could be enhanced with LLM
        # For now, return as-is
        return original_query
    
    async def _arun(
        self,
        query: str,
        k: int = 5,
        rewrite_query: bool = True
    ) -> Dict[str, Any]:
        """Async version of _run"""
        return self._run(query, k, rewrite_query)


def create_vector_db_tool() -> VectorDBTool:
    """Factory function to create VectorDBTool instance"""
    return VectorDBTool()
