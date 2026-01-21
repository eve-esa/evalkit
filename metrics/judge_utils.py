"""
Centralized LLM Judge utilities for evaluation tasks.

This module provides reusable components for LLM-based evaluation:
- LoggableFuture: Wrapper for async judge evaluations
- QA Judge: Scoring for question-answering tasks
- Common utilities for client initialization and result processing
"""

import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union

import yaml
from openai import OpenAI

# Disable verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)


# Default parallelism, can be overridden via JUDGE_MAX_PARALLEL env var
MAX_PARALLEL_REQUESTS = int(os.getenv("JUDGE_MAX_PARALLEL", "5"))

# Delay between judge API requests (seconds) to avoid rate limiting
JUDGE_REQUEST_DELAY = float(os.getenv("JUDGE_REQUEST_DELAY", "0.0"))

# Default score when judge fails (use extreme negative to make failures obvious)
JUDGE_FAILURE_SCORE = -10000

# Global state for lazy initialization
_JUDGE_CLIENT = None
_QA_PROMPT_TEMPLATE = None
_QA_PROMPT_TEMPLATE_DEFAULT = None
_EXECUTOR = None


def get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for judge calls."""
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS)
    return _EXECUTOR


def reset_judge_state():
    """
    Reset all global state for the judge module.

    Call this between evaluations if judge configuration changes
    (different API keys, base URLs, or model names).
    """
    global _JUDGE_CLIENT, _EXECUTOR
    _JUDGE_CLIENT = None
    _EXECUTOR = None


def shutdown_executor(wait: bool = True):
    """
    Shutdown the thread pool executor.

    Args:
        wait: If True, wait for all pending tasks to complete.
    """
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=wait)
        _EXECUTOR = None


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
    Initializes and returns a singleton OpenAI client for judge calls.

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

    return _JUDGE_CLIENT


# ============================================================================
# Generic Judge Functions
# ============================================================================


def _judge_qa_generic(
    sample: dict,
    prompt_getter,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict:
    """Generic judge function for QA evaluation with 0-5 scoring."""
    if model_name is None:
        model_name = os.getenv("JUDGE_NAME") or os.getenv("JUDGE_MODEL") or "mistral-medium-latest"

    client = get_judge_client(api_key=api_key, base_url=base_url)
    prompt_template = prompt_getter()

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
                        "enum": [0, 1, 2, 3, 4, 5],
                        "description": "Score from 0 (fail) to 5 (excellent)",
                    }
                },
                "required": ["score"],
                "additionalProperties": False,
            },
        },
    }

    simple_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4, 5],
                "description": "Score from 0 (fail) to 5 (excellent).",
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
        if JUDGE_REQUEST_DELAY > 0:
            time.sleep(JUDGE_REQUEST_DELAY)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": final_prompt}],
                response_format=json_schema,
                temperature=0.0,
                max_tokens=100,
            )
        except Exception:
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

        if score in [0, 1, 2, 3, 4, 5]:
            return {"score": int(score), "raw_output": response_content}
        else:
            return {"score": JUDGE_FAILURE_SCORE, "raw_output": response_content}

    except Exception as e:
        return {"score": JUDGE_FAILURE_SCORE, "raw_output": f"Error: {str(e)}"}


def _process_qa_results_generic(
    doc, results, judge_func, metric_name, question_key="question", answer_key="answer", **kwargs
):
    """Generic process function for QA results."""
    if "Question" in doc:
        question_key = "Question"
        answer_key = "Answer"
    elif "question" in doc:
        question_key = "question"
        answer_key = "answer"

    sample = {
        "question": doc[question_key],
        "output": results[0],
        "reference": doc[answer_key],
    }

    executor = get_executor()
    future = executor.submit(judge_func, sample)

    return {metric_name: LoggableFuture(future)}


def _aggregate_llm_judge_generic(items: Union[List[LoggableFuture], List[Future]]) -> float:
    """
    Aggregate LLM judge results (0-5 scale) by waiting for futures and calculating mean.

    The result is normalized to 0-1 scale (divide by 5) for compatibility with other metrics.

    Args:
        items: List of LoggableFuture or Future objects containing score dictionaries.

    Returns:
        Mean score across all items normalized to 0-1 scale, or 0.0 if items is empty.
    """
    if not items:
        return 0.0

    scores = []
    for item in items:
        result = item.result() if isinstance(item, (LoggableFuture, Future)) else item
        if isinstance(result, dict):
            scores.append(result["score"])
        else:
            scores.append(result)

    raw_mean = mean(scores) if scores else 0.0
    return raw_mean / 5.0


# ============================================================================
# Default Judge Functions (penalty-based acronym check)
# ============================================================================


def get_qa_prompt_template_default() -> str:
    """Load and return the default QA judge prompt template."""
    global _QA_PROMPT_TEMPLATE_DEFAULT

    if _QA_PROMPT_TEMPLATE_DEFAULT is None:
        prompt_path = Path(__file__).parent / "prompts" / "llm_judge_qa_default.yaml"
        config = yaml.safe_load(prompt_path.read_text())
        _QA_PROMPT_TEMPLATE_DEFAULT = config["prompt"]

    return _QA_PROMPT_TEMPLATE_DEFAULT


def judge_qa_with_llm_default(sample: dict, **kwargs) -> Dict:
    """Default QA judge with penalty-based acronym check."""
    return _judge_qa_generic(sample, get_qa_prompt_template_default, **kwargs)


def process_qa_results_default(doc, results, **kwargs):
    """Process QA results with the default LLM judge."""
    return _process_qa_results_generic(
        doc, results, judge_qa_with_llm_default, "llm_as_judge_default", **kwargs
    )


aggregate_llm_judge_default = _aggregate_llm_judge_generic
