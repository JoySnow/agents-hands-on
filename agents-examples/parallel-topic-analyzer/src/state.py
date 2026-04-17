"""State schema for the topic analysis agent."""

from typing import TypedDict


class TopicAnalysisState(TypedDict, total=False):
    """State for the topic analysis workflow.

    Attributes:
        topic: The input topic to analyze
        summary: Concise summary of the topic
        questions: Three interesting questions about the topic
        key_terms: 5-10 key terms from the topic (comma-separated)
        final_result: Combined structured output with metadata
        errors: List of errors encountered during processing
    """
    topic: str
    summary: str
    questions: str
    key_terms: str
    final_result: dict
    errors: list[str]
