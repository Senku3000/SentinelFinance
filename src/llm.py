"""LLM provider factory."""

from langchain_groq import ChatGroq

from .config import Config


def create_chat_llm(temperature: float = 0.3):
    """Create the configured chat model.

    Set LLM_PROVIDER=groq for the hosted Groq API or LLM_PROVIDER=ollama for a
    local Ollama model.
    """
    if Config.LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Ollama provider selected, but langchain-ollama is not installed. "
                "Run: pip install langchain-ollama"
            ) from exc

        return ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    return ChatGroq(
        model=Config.GROQ_MODEL,
        groq_api_key=Config.GROQ_API_KEY,
        temperature=temperature,
    )
