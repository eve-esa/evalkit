"""
Centralized LLM Judge utilities for evaluation tasks.

This module provides reusable components for LLM-based evaluation:
- LoggableFuture: Wrapper for async judge evaluations
- QA Judge: Binary scoring (0/1) for question-answering tasks
- Summary Judge: Multi-dimensional scoring for summarization tasks
- Common utilities for client initialization and result processing
"""

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union

import yaml
from openai import OpenAI


MAX_PARALLEL_REQUESTS = 10

# Global state for lazy initialization
_JUDGE_CLIENT = None
_QA_PROMPT_TEMPLATE = None
_SUMMARY_PROMPT_TEMPLATE = None
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS)


class LoggableFuture:
    """Wrapper around Future that provides a nicer string representation for logging."""

    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        """Get the result from the underlying future."""
        return self._future.result(timeout=timeout)

    def __repr__(self):
        """Return a nicer representation for logging."""
        if self._future.done():
            try:
                return str(self._future.result())
            except Exception as e:
                return f"<Failed: {type(e).__name__}>"
        else:
            return "<Pending LLM judge evaluation>"

    def __str__(self):
        return self.__repr__()


def get_judge_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Initializes and returns a singleton litellm client.

    Args:
        api_key: API key for the judge model. If None, reads from environment.
        base_url: Base URL for the API. If None, reads from environment or uses default.

    Returns:
        Configured OpenAI client instance.

    Raises:
        ValueError: If no API key is found.
    """
    global _JUDGE_CLIENT

    if _JUDGE_CLIENT is None:
        # Try multiple environment variables for API key
        if api_key is None:
            api_key = (
                os.getenv("JUDGE_API_KEY")
                or os.getenv("MISTRAL_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )

        if not api_key:
            raise ValueError(
                "No API key found. Set JUDGE_API_KEY, MISTRAL_API_KEY, "
                "or OPENAI_API_KEY environment variable."
            )

        # Determine base URL
        if base_url is None:
            base_url = os.getenv("JUDGE_BASE_URL") or "https://api.mistral.ai/v1"

        _JUDGE_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
        print(f"Initialized LLM Judge client with base URL: {base_url}")

    return _JUDGE_CLIENT


def get_qa_prompt_template() -> str:
    """Load and return the QA judge prompt template."""
    global _QA_PROMPT_TEMPLATE

    if _QA_PROMPT_TEMPLATE is None:
        prompt_path = Path(__file__).parent / "prompts/llm_judge_qa.yaml"
        config = yaml.safe_load(prompt_path.read_text())
        _QA_PROMPT_TEMPLATE = config["prompt"]

    return _QA_PROMPT_TEMPLATE


def judge_qa_with_llm(
    sample: dict,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> int:
    """
    Calls the LLM judge for QA evaluation with binary (0/1) scoring.

    Args:
        sample: Dictionary with "question", "output", and "reference" keys.
        model_name: Name of the judge model. If None, reads from JUDGE_NAME env var.
        api_key: API key for the judge. If None, uses environment variables.
        base_url: Base URL for the API. If None, uses environment variables.

    Returns:
        Integer score: 1 for correct, 0 for incorrect.
    """
    if model_name is None:
        model_name = os.getenv("JUDGE_NAME") or os.getenv("JUDGE_MODEL") or "mistral-large-latest"

    client = get_judge_client(api_key=api_key, base_url=base_url)
    prompt_template = get_qa_prompt_template()

    # JSON schema for structured output (OpenAI-style)
    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "enum": [0, 1],
                        "description": "1 if correct, 0 if incorrect",
                    }
                },
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }

    # Simple schema for format instructions
    simple_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "enum": [0, 1],
                "description": "1 if correct, 0 if incorrect.",
            }
        },
        "required": ["score"],
    }

    format_instructions = (
        "You must respond with a JSON object that strictly follows this schema:\n"
        f"{json.dumps(simple_schema, indent=2)}"
    )

    final_prompt = prompt_template.format(
        question=sample["question"],
        output=sample["output"],
        reference=sample["reference"],
        format_instructions=format_instructions,
    )

    try:
        # Try structured outputs first (OpenAI-compatible APIs)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": final_prompt}],
                response_format=json_schema,
                temperature=0.0,
                max_tokens=100,
            )
        except Exception:
            # Fall back to basic JSON mode if structured outputs not supported
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": final_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100,
            )

        response_content = response.choices[0].message.content
        if response_content is None:
            raise ValueError("LLM response content is None")

        data = json.loads(response_content)
        score = data.get("score")

        if score in [0, 1]:
            return int(score)
        else:
            print(f"Warning: Judge returned invalid score: {score}. Defaulting to 0.")
            print(f"Full response: {response_content}")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Output: {sample['output'][:100]}...")
            return 0

    except Exception as e:
        print(f"Error during LLM judge call: {e}. Defaulting to score 0.")
        print(f"Question: {sample['question'][:100]}...")
        return 0


def process_qa_results(
    doc: dict,
    results: list[str],
    question_key: str = "question",
    answer_key: str = "answer",
    sleep_time: float = 0.0,
) -> dict:
    """
    Process QA results with LLM judge in background thread.

    Args:
        doc: Document dictionary containing the question and reference answer.
        results: List of model outputs (first element is used).
        question_key: Key in doc for the question text.
        answer_key: Key in doc for the reference answer.
        sleep_time: Optional delay before submitting (for rate limiting).

    Returns:
        Dictionary with "llm_as_judge" key containing LoggableFuture.
    """
    sample = {
        "question": doc[question_key],
        "output": results[0],
        "reference": doc[answer_key],
    }

    future = _EXECUTOR.submit(judge_qa_with_llm, sample)

    if sleep_time > 0:
        time.sleep(sleep_time)

    return {"llm_as_judge": LoggableFuture(future)}


def aggregate_llm_judge(items: Union[List[LoggableFuture], List[Future]]) -> float:
    """
    Aggregate LLM judge results by waiting for futures and calculating mean.

    Args:
        items: List of LoggableFuture or Future objects containing scores.

    Returns:
        Mean score across all items, or 0.0 if items is empty.
    """
    if not items:
        return 0.0

    # Handle both LoggableFuture and regular Future objects
    scores = []
    for item in items:
        if isinstance(item, LoggableFuture):
            scores.append(item.result())
        else:
            scores.append(item.result())

    return mean(scores) if scores else 0.0
