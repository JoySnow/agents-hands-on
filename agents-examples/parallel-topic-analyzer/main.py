#!/usr/bin/env python3
"""CLI entry point for the parallel topic analyzer agent."""

import argparse
import json
import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv

from src.agent import TopicAnalyzerAgent


def check_ollama_server(base_url: str) -> bool:
    """Check if Ollama server is reachable.

    Args:
        base_url: Ollama server URL

    Returns:
        True if server is reachable, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/version", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_model_available(base_url: str, model: str) -> bool:
    """Check if a model is available on the Ollama server.

    Args:
        base_url: Ollama server URL
        model: Model name to check

    Returns:
        True if model is available, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            model_names = [m["name"] for m in models]
            return any(model in name or name.startswith(model + ":") for name in model_names)
        return False
    except requests.RequestException:
        return False


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Analyze a topic using parallel LLM tasks with LangGraph"
    )
    parser.add_argument(
        "topic",
        type=str,
        help="The topic to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b"),
        help="Ollama model to use (default: deepseek-r1:1.5b)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with timing information"
    )

    args = parser.parse_args()

    if not args.topic.strip():
        print("Error: Topic cannot be empty", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Checking Ollama server at {args.ollama_host}...")

    if not check_ollama_server(args.ollama_host):
        print(f"Error: Cannot connect to Ollama server at {args.ollama_host}", file=sys.stderr)
        print("Please ensure Ollama is running with: ollama serve", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Checking if model '{args.model}' is available...")

    if not check_model_available(args.ollama_host, args.model):
        print(f"Error: Model '{args.model}' not found", file=sys.stderr)
        print(f"Pull it with: ollama pull {args.model}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Analyzing topic: {args.topic}")
        print(f"Using model: {args.model}")
        print("Executing parallel tasks...")

    try:
        agent = TopicAnalyzerAgent(
            ollama_base_url=args.ollama_host,
            model=args.model
        )

        result = agent.analyze(args.topic)

        if args.verbose:
            exec_time = result.get("metadata", {}).get("execution_time_seconds", 0)
            print(f"Completed in {exec_time}s\n")

        print(json.dumps(result, indent=2, ensure_ascii=False))

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
