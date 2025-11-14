import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import yaml
from litellm import OpenAI

MAX_PARALLEL_REQUESTS = 10


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
                return f"{self._future.result()}"
            except Exception as e:
                return f"<Failed: {type(e).__name__}>"
        else:
            return "<Pending LLM judge evaluation>"

    def __str__(self):
        return self.__repr__()


PROMPT_PATH = Path(__file__).parent / "../prompts/llm_judge_qa.yaml"
JUDGE_MODEL = os.getenv("JUDGE_NAME") or os.getenv("JUDGE_MODEL") or "mistral-large-latest"


# Lazy init
_JUDGE_CLIENT = None
_PROMPT_TEMPLATE = None
# background threads for parallel API calls.
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS)


def get_judge_client_and_prompt():
    """Initializes and returns a singleton litellm client and the prompt template.

    Prevents re-creating the client for every single evaluation.
    """
    global _JUDGE_CLIENT, _PROMPT_TEMPLATE

    if _JUDGE_CLIENT is None:
        api_key = os.getenv("JUDGE_API_KEY") or os.getenv("MISTRAL_API_KEY")
        base_url = os.getenv("JUDGE_BASE_URL") or "https://api.mistral.ai/v1"
        if not api_key:
            raise ValueError("JUDGE_API_KEY or MISTRAL_API_KEY environment variable not set.")

        _JUDGE_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
        print("Initialized LLM Judge client.")

    if _PROMPT_TEMPLATE is None:
        config = yaml.safe_load(Path(PROMPT_PATH).read_text())
        _PROMPT_TEMPLATE = config["prompt"]

    return _JUDGE_CLIENT, _PROMPT_TEMPLATE


def judge_with_llm(sample: dict) -> int:
    """Calls the LLM with a specific sample and forces a 0 or 1 JSON response.
    Args:
        sample: A dictionary with "question", "output", and "reference".

    Returns:
        An integer score: 1 for correct, 0 for incorrect.
    """
    client, prompt_template = get_judge_client_and_prompt()

    # JSON schema for structured output
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
                        "description": "1 if correct, 0 if incorrect"
                    }
                },
                "required": ["score"],
                "additionalProperties": False
            }
        }
    }

    # Format the prompt (no format_instructions needed with structured output)
    final_prompt = prompt_template.format(
        question=sample["question"],
        output=sample["output"],
        reference=sample["reference"],
        format_instructions="Respond with a JSON object containing a 'score' field with value 0 or 1."
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            response_format=json_schema,
            temperature=0.0,
            max_tokens=100,
        )

        response_content = response.choices[0].message.content
        if response_content is None:
            raise ValueError("LLM response content is None")
        data = json.loads(response_content)

        # Extract the score (guaranteed by structured output to be 0 or 1)
        score = data.get("score")
        if score in [0, 1]:
            return int(score)
        else:
            print(
                f"Warning: Judge returned an invalid score: {score}. Defaulting to 0."
            )
            print(f"Full response: {response_content}")
            print(f"Question: {sample['question'][:100]}...")
            print(f"Output: {sample['output'][:100]}...")
            return 0

    except Exception as e:
        # If the API call fails or JSON is malformed, score as incorrect.
        print(f"Error during LLM judge call: {e}. Defaulting to score 0.")
        return 0


# without //threading
# # def process_results(doc: dict, results: list[str]) -> dict:
#     sample = {
#         "question": doc["question"],
#         "output": results[0],
#         "reference": doc["answer"],
#     }
#     judge_score = judge_with_llm(sample)
#     return {"llm_judge": judge_score}


def process_results(doc: dict, results: list[str]) -> dict:
    sample = {
        "question": doc["question"],
        "output": results[0],
        "reference": doc["answer"],
    }

    future = _EXECUTOR.submit(judge_with_llm, sample)
    return {"llm_judge_future": LoggableFuture(future)}


def process_results2(doc: dict, results: list[str]) -> dict:
    sample = {
        "question": doc["Question"],
        "output": results[0],
        "reference": doc["Answer"],
    }
    future = _EXECUTOR.submit(judge_with_llm, sample)
    time.sleep(0.2)
    return {"llm_judge_future": LoggableFuture(future)}


def aggregate_llm_judge(items: list[LoggableFuture]) -> float:
    """Waits for all the future results to complete and calculates the mean score."""
    scores = [future.result() for future in items]
    return sum(scores) / len(scores) if scores else 0
