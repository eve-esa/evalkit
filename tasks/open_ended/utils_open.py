import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import yaml
from litellm import OpenAI

MAX_PARALLEL_REQUESTS = 10


PROMPT_PATH = Path(__file__).parent / "../prompts/llm_judge_qa.yaml"
JUDGE_MODEL = "mistral-large-latest"


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
        api_key = os.getenv("MISTRAL_API_KEY")
        base_url = os.getenv("MISTRAL_BASE_URL")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")

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

    # JSON schema for the LLM to follow.
    json_schema = {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "enum": [0, 1],
                "description": "1 if correct, 0 if it is incorrect.",
            }
        },
        "required": ["score"],
    }

    # instructions to be injected into the prompt.
    format_instructions = (
        "You must respond with a JSON object that strictly follows this schema:\n"
        f"{json.dumps(json_schema, indent=2)}"
    )

    # formatting
    final_prompt = prompt_template.format(
        question=sample["question"],
        output=sample["output"],
        reference=sample["reference"],
        format_instructions=format_instructions,
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"},  # Force JSON mode
            temperature=0.0,
            max_tokens=100,
        )

        response_content = response.choices[0].message.content
        if response_content is None:
            raise ValueError("LLM response content is None")
        data = json.loads(response_content)

        # Safely extract the score. Default to 0 if something is wrong.
        score = data.get("score")
        if score in [0, 1]:
            return int(score)
        else:
            print(
                f"Warning: Judge returned an invalid score: {score}. Defaulting to 0."
            )
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
    return {"llm_judge_future": future}


def process_results2(doc: dict, results: list[str]) -> dict:
    sample = {
        "question": doc["Question"],
        "output": results[0],
        "reference": doc["Answer"],
    }
    future = _EXECUTOR.submit(judge_with_llm, sample)
    return {"llm_judge_future": future}


def aggregate_llm_judge(items: list[Future]) -> float:
    """Waits for all the future results to complete and calculates the mean score."""
    scores = [future.result() for future in items]
    return sum(scores) / len(scores) if scores else 0
