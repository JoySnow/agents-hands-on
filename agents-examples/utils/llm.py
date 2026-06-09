from langchain_ollama import ChatOllama


def ollama_create_llm(
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:1.5b",
        temperature: str = "0.7") -> ChatOllama:
    """Create a ChatOllama instance.

    Args:
        base_url: Ollama server URL
        model: Model name to use

    Returns:
        Configured ChatOllama instance
    """
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
