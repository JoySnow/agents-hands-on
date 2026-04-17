"""Task nodes for parallel execution."""

import re
from datetime import datetime

from langchain_ollama import ChatOllama

from .state import TopicAnalysisState


def create_llm(base_url: str, model: str) -> ChatOllama:
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
        temperature=0.7,
    )


def summarize_task(state: TopicAnalysisState, llm: ChatOllama) -> dict:
    """Task A: Summarize the topic concisely.

    Args:
        state: Current workflow state
        llm: ChatOllama instance

    Returns:
        State update with summary only
    """
    try:
        prompt = f"Summarize the following topic concisely: {state['topic']}"
        response = llm.invoke(prompt)
        summary = _strip_thinking_tags(response.content.strip())
        return {"summary": summary}
    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"Summarize task failed: {str(e)}")
        return {"summary": "", "errors": errors}


def questions_task(state: TopicAnalysisState, llm: ChatOllama) -> dict:
    """Task B: Generate three interesting questions about the topic.

    Args:
        state: Current workflow state
        llm: ChatOllama instance

    Returns:
        State update with questions only
    """
    try:
        prompt = f"Generate three interesting questions about the following topic: {state['topic']}"
        response = llm.invoke(prompt)
        return {"questions": response.content.strip()}
    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"Questions task failed: {str(e)}")
        return {"questions": "", "errors": errors}


def key_terms_task(state: TopicAnalysisState, llm: ChatOllama) -> dict:
    """Task C: Identify 5-10 key terms from the topic.

    Args:
        state: Current workflow state
        llm: ChatOllama instance

    Returns:
        State update with key terms only
    """
    try:
        prompt = f"Identify 5-10 key terms from the following topic, separated by commas: {state['topic']}"
        response = llm.invoke(prompt)
        return {"key_terms": response.content.strip()}
    except Exception as e:
        errors = state.get("errors", [])
        errors.append(f"Key terms task failed: {str(e)}")
        return {"key_terms": "", "errors": errors}


def _strip_thinking_tags(text: str) -> str:
    """Remove <think> tags and their content from model output.

    Args:
        text: Raw text from LLM

    Returns:
        Cleaned text without thinking process
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def _parse_questions(questions_text: str) -> list[str]:
    """Parse questions from LLM response into a list.

    Args:
        questions_text: Raw questions text from LLM

    Returns:
        List of parsed questions (up to 3)
    """
    if not questions_text:
        return []

    questions_text = _strip_thinking_tags(questions_text)

    lines = questions_text.strip().split('\n')
    parsed = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\d+[\.)]\s*', '', line)
        line = re.sub(r'^[-*•]\s*', '', line)
        if line and '?' in line:
            parsed.append(line)

    return parsed[:3]


def _parse_key_terms(key_terms_text: str) -> list[str]:
    """Parse key terms from LLM response into a list.

    Args:
        key_terms_text: Raw key terms text from LLM

    Returns:
        List of parsed key terms (5-10)
    """
    if not key_terms_text:
        return []

    key_terms_text = _strip_thinking_tags(key_terms_text)

    terms = [term.strip() for term in key_terms_text.split(',')]
    terms = [term for term in terms if term and len(term) > 2 and not term.startswith('<')]

    return terms[:10]


def combine_results(state: TopicAnalysisState, start_time: float) -> dict:
    """Combine all task outputs into final structured result.

    Args:
        state: Current workflow state with all task results
        start_time: Timestamp when processing started

    Returns:
        State update with final_result only
    """
    import time

    execution_time = time.time() - start_time

    questions_list = _parse_questions(state.get("questions", ""))
    key_terms_list = _parse_key_terms(state.get("key_terms", ""))

    final_result = {
        "topic": state["topic"],
        "results": {
            "summary": state.get("summary", ""),
            "questions": questions_list,
            "key_terms": key_terms_list,
        },
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "execution_time_seconds": round(execution_time, 2),
            "parallel_execution": True,
        }
    }

    if state.get("errors"):
        final_result["metadata"]["errors"] = state["errors"]

    return {"final_result": final_result}
