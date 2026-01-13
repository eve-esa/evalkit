"""
Centralized LLM Judge utilities for evaluation tasks.

This module provides reusable components for LLM-based evaluation:
- LoggableFuture: Wrapper for async judge evaluations
- QA Judge: Binary scoring (0/1) for question-answering tasks
- Summary Judge: Multi-dimensional scoring for summarization tasks
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


MAX_PARALLEL_REQUESTS = 10

# Global state for lazy initialization
_JUDGE_CLIENTS = {}  # Changed to dict to support multiple clients
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


class LoggableFutureExtractor:
    """Wrapper that extracts a specific judge's result from multi-judge future."""

    def __init__(self, future: Future, judge_name: str):
        self._future = future
        self._judge_name = judge_name

    def result(self, timeout=None):
        """Extract the specific judge's result from the multi-judge result."""
        multi_result = self._future.result(timeout=timeout)
        if isinstance(multi_result, dict) and self._judge_name in multi_result:
            return multi_result[self._judge_name]
        return {"score": 0, "raw_output": "Judge not found"}

    def __repr__(self):
        """Return a nicer representation for logging."""
        if self._future.done():
            try:
                result = self.result()
                if isinstance(result, dict) and "score" in result:
                    return str(result["score"])
                return str(result)
            except Exception as e:
                return f"<Failed: {type(e).__name__}>"
        else:
            return f"<Pending {self._judge_name} evaluation>"

    def __str__(self):
        return self.__repr__()


class LoggableFutureAgreement:
    """Wrapper that calculates agreement from multi-judge future."""

    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        """Calculate agreement score (1 if all agree, 0 otherwise) from multi-judge result."""
        multi_result = self._future.result(timeout=timeout)
        if isinstance(multi_result, dict):
            scores = [v["score"] for v in multi_result.values() if isinstance(v, dict) and "score" in v]
            if scores:
                return 1 if len(set(scores)) == 1 else 0
        return 0

    def __repr__(self):
        """Return a nicer representation for logging."""
        if self._future.done():
            try:
                return str(self.result())
            except Exception as e:
                return f"<Failed: {type(e).__name__}>"
        else:
            return "<Pending agreement calculation>"

    def __str__(self):
        return self.__repr__()


class LoggableFutureVoting:
    """Wrapper that calculates majority voting result from multi-judge future."""

    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        """Calculate majority vote (returns majority score if >= half+1 judges agree, 0 otherwise)."""
        multi_result = self._future.result(timeout=timeout)
        if isinstance(multi_result, dict):
            scores = [v["score"] for v in multi_result.values() if isinstance(v, dict) and "score" in v]
            if scores:
                # Count votes for each score
                from collections import Counter
                vote_counts = Counter(scores)

                # Calculate majority threshold (half + 1)
                majority_threshold = (len(scores) // 2) + 1

                # Get the most common score and its count
                most_common_score, count = vote_counts.most_common(1)[0]

                # Return the most common score if it meets the majority threshold
                if count >= majority_threshold:
                    return most_common_score

                # No majority reached, return 0
                return 0
        return 0

    def __repr__(self):
        """Return a nicer representation for logging."""
        if self._future.done():
            try:
                return str(self.result())
            except Exception as e:
                return f"<Failed: {type(e).__name__}>"
        else:
            return "<Pending voting calculation>"

    def __str__(self):
        return self.__repr__()


def get_judge_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """
    Initializes and returns a cached OpenAI client based on credentials.

    Args:
        api_key: API key for the judge model. If None, reads from environment.
        base_url: Base URL for the API. If None, reads from environment or uses default.

    Returns:
        Configured OpenAI client instance.

    Raises:
        ValueError: If no API key is found.
    """
    global _JUDGE_CLIENTS

    # Try multiple environment variables for API key if not provided
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

    # Create cache key from credentials
    cache_key = f"{api_key[:10]}:{base_url}"

    # Return cached client or create new one
    if cache_key not in _JUDGE_CLIENTS:
        _JUDGE_CLIENTS[cache_key] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,  # 60 second timeout to prevent hanging
            max_retries=2
        )

    return _JUDGE_CLIENTS[cache_key]


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
) -> Dict:
    """
    Calls the LLM judge for QA evaluation with binary (0/1) scoring.

    Args:
        sample: Dictionary with "question", "output", and "reference" keys.
        model_name: Name of the judge model. If None, reads from JUDGE_NAME env var.
        api_key: API key for the judge. If None, uses environment variables.
        base_url: Base URL for the API. If None, uses environment variables.

    Returns:
        Dictionary with "score" (int) and "raw_output" (str) keys.
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
            return {"score": int(score), "raw_output": response_content}
        else:
            print(f"Warning: Judge returned invalid score: {score}. Defaulting to 0.")
            print(f"Full response: {response_content}")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Output: {sample['output'][:100]}...")
            return {"score": 0, "raw_output": response_content}

    except Exception as e:
        print(f"Error during LLM judge call: {e}. Defaulting to score 0.")
        print(f"Question: {sample['question'][:100]}...")
        error_msg = f"Error: {str(e)}"
        return {"score": 0, "raw_output": error_msg}


def judge_qa_with_multiple_llms(
    sample: dict,
    judges: List[Dict],
) -> Dict[str, Dict]:
    """
    Calls multiple LLM judges for QA evaluation.

    Args:
        sample: Dictionary with "question", "output", and "reference" keys.
        judges: List of judge configurations, each with optional keys:
            - name: Name identifier for the judge
            - model: Model name to use
            - api_key: API key for the judge
            - base_url: Base URL for the API

    Returns:
        Dictionary mapping judge names to their result dicts (score + raw_output).
    """
    question_preview = sample.get("question", "")[:80]
    print(f"[DEBUG multi-judge] Starting evaluation for: {question_preview}...")

    results = {}
    for i, judge_config in enumerate(judges, 1):
        judge_name = judge_config.get("name", judge_config.get("model", "unknown"))
        model_name = judge_config.get("model")
        api_key = judge_config.get("api_key")
        base_url = judge_config.get("base_url")

        print(f"[DEBUG multi-judge] Judge {i}/{len(judges)}: {judge_name} (model: {model_name})")
        try:
            result = judge_qa_with_llm(
                sample=sample,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            print(f"[DEBUG multi-judge] Judge {judge_name} completed: score={result.get('score', 'N/A')}")
            results[judge_name] = result
        except Exception as e:
            print(f"[ERROR multi-judge] Judge {judge_name} failed: {e}")
            results[judge_name] = {"score": 0, "raw_output": f"Error: {str(e)}"}

    print(f"[DEBUG multi-judge] All judges completed")
    return results


def process_qa_results(
    doc: dict,
    results: list[str],
    question_key: str = "question",
    answer_key: str = "answer",
    sleep_time: float = 0.0,
    judges: Optional[List[Dict]] = None,
) -> dict:
    """
    Process QA results with LLM judge(s) in background thread.

    Args:
        doc: Document dictionary containing the question and reference answer.
        results: List of model outputs (first element is used).
        question_key: Key in doc for the question text.
        answer_key: Key in doc for the reference answer.
        sleep_time: Optional delay before submitting (for rate limiting).
        judges: Optional list of judge configurations for multi-judge evaluation.
            If None, uses single judge from environment variables.

    Returns:
        Dictionary with judge results. If multiple judges, creates separate keys:
            {
                "llm_as_judge_<name>": LoggableFuture for each judge,
                "llm_as_judge_multi": LoggableFuture with all results,
                "judge_agreement": LoggableFuture with agreement data (1 if all agree, 0 otherwise),
                "judge_voting": LoggableFuture with majority vote result (returns majority score if >= half+1 judges agree)
            }
        If single judge:
            {"llm_as_judge": LoggableFuture}
    """
    sample = {
        "question": doc[question_key],
        "output": results[0],
        "reference": doc[answer_key],
    }

    if judges and len(judges) > 0:
        # Multi-judge mode
        future = _EXECUTOR.submit(judge_qa_with_multiple_llms, sample, judges)

        if sleep_time > 0:
            time.sleep(sleep_time)

        # Create separate futures for each judge that will extract their specific results
        result_dict = {}

        # Store the complete multi-judge result
        result_dict["llm_as_judge_multi"] = LoggableFuture(future)

        # Create individual judge metrics
        for judge_config in judges:
            judge_name = judge_config.get("name", judge_config.get("model", "unknown"))
            # Create a future wrapper that extracts this specific judge's result
            result_dict[f"llm_as_judge_{judge_name}"] = LoggableFutureExtractor(future, judge_name)

        # Add agreement metric (all judges must agree)
        result_dict["judge_agreement"] = LoggableFutureAgreement(future)

        # Add voting metric (majority vote)
        result_dict["judge_voting"] = LoggableFutureVoting(future)

        return result_dict
    else:
        # Single judge mode (backward compatibility)
        future = _EXECUTOR.submit(judge_qa_with_llm, sample)

        if sleep_time > 0:
            time.sleep(sleep_time)

        return {"llm_as_judge": LoggableFuture(future)}


def aggregate_llm_judge(items: Union[List[LoggableFuture], List[Future], List[LoggableFutureExtractor], List[LoggableFutureAgreement], List[LoggableFutureVoting]]) -> float:
    """
    Aggregate LLM judge results by waiting for futures and calculating mean.

    Args:
        items: List of LoggableFuture or Future objects containing score dictionaries.

    Returns:
        Mean score across all items, or 0.0 if items is empty.
    """
    if not items:
        return 0.0

    # Handle both LoggableFuture and regular Future objects
    scores = []
    for item in items:
        # Get result from any future-like object
        if hasattr(item, 'result'):
            result = item.result()
        else:
            result = item

        # Handle both dict format (new) and int format (backward compatibility)
        if isinstance(result, dict):
            # Check if this is multi-judge result
            if all(isinstance(v, dict) and "score" in v for v in result.values()):
                # Multi-judge: average across all judges for this sample
                judge_scores = [v["score"] for v in result.values()]
                scores.append(mean(judge_scores))
            elif "score" in result:
                # Single judge dict format
                scores.append(result["score"])
        elif isinstance(result, (int, float)):
            scores.append(result)

    return mean(scores) if scores else 0.0


def create_judge_aggregator(judge_name: str):
    """
    Factory function to create aggregation functions for specific judges.

    Args:
        judge_name: Name of the judge to aggregate results for.

    Returns:
        Aggregation function for that specific judge.
    """
    def aggregate_specific_judge(items: Union[List[LoggableFuture], List[Future]]) -> float:
        if not items:
            return 0.0

        scores = []
        for item in items:
            result = item.result() if isinstance(item, (LoggableFuture, Future)) else item

            if isinstance(result, dict):
                # Multi-judge format
                if judge_name in result and isinstance(result[judge_name], dict):
                    scores.append(result[judge_name]["score"])
                # Single judge format
                elif "score" in result:
                    scores.append(result["score"])

        return mean(scores) if scores else 0.0

    return aggregate_specific_judge


def calculate_judge_agreement(
    items: Union[List[LoggableFuture], List[Future]]
) -> float:
    """
    Calculate agreement rate between multiple judges.

    Args:
        items: List of LoggableFuture or Future objects containing multi-judge results.

    Returns:
        Agreement rate (0.0 to 1.0), or 0.0 if not applicable.
    """
    if not items:
        return 0.0

    agreement_count = 0
    total_count = 0

    for item in items:
        result = item.result() if isinstance(item, (LoggableFuture, Future)) else item

        if isinstance(result, dict):
            # Check if multi-judge format
            if all(isinstance(v, dict) and "score" in v for v in result.values()):
                scores = [v["score"] for v in result.values()]
                # All judges agree if all scores are the same
                if len(set(scores)) == 1:
                    agreement_count += 1
                total_count += 1

    return agreement_count / total_count if total_count > 0 else 0.0
