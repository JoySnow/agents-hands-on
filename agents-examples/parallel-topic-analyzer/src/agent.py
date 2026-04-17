"""LangGraph agent with parallel task execution."""

import time

from langgraph.graph import StateGraph, START, END

from .state import TopicAnalysisState
from .tasks import (
    create_llm,
    summarize_task,
    questions_task,
    key_terms_task,
    combine_results,
)


class TopicAnalyzerAgent:
    """Agent for analyzing topics with parallel LLM tasks.

    This agent uses LangGraph's StateGraph to execute three analysis tasks
    in parallel (summarize, generate questions, extract key terms), then
    combines the results into a structured output.
    """

    def __init__(self, ollama_base_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        """Initialize the topic analyzer agent.

        Args:
            ollama_base_url: URL of the Ollama server
            model: Name of the model to use
        """
        self.ollama_base_url = ollama_base_url
        self.model = model
        self.llm = create_llm(ollama_base_url, model)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with parallel execution.

        Returns:
            Compiled state graph
        """
        graph = StateGraph(TopicAnalysisState)

        def summarize_node(state: TopicAnalysisState) -> TopicAnalysisState:
            return summarize_task(state, self.llm)

        def questions_node(state: TopicAnalysisState) -> TopicAnalysisState:
            return questions_task(state, self.llm)

        def key_terms_node(state: TopicAnalysisState) -> TopicAnalysisState:
            return key_terms_task(state, self.llm)

        def combine_node(state: TopicAnalysisState) -> TopicAnalysisState:
            return combine_results(state, self._start_time)

        graph.add_node("summarize", summarize_node)
        graph.add_node("questions", questions_node)
        graph.add_node("key_terms", key_terms_node)
        graph.add_node("combine", combine_node)

        graph.add_edge(START, "summarize")
        graph.add_edge(START, "questions")
        graph.add_edge(START, "key_terms")

        graph.add_edge("summarize", "combine")
        graph.add_edge("questions", "combine")
        graph.add_edge("key_terms", "combine")

        graph.add_edge("combine", END)

        return graph.compile()

    def analyze(self, topic: str) -> dict:
        """Analyze a topic using parallel LLM tasks.

        Args:
            topic: The topic to analyze

        Returns:
            Dictionary containing analysis results and metadata
        """
        self._start_time = time.time()

        initial_state: TopicAnalysisState = {
            "topic": topic,
            "summary": "",
            "questions": "",
            "key_terms": "",
            "errors": [],
        }

        final_state = self.graph.invoke(initial_state)

        return final_state.get("final_result", {})
