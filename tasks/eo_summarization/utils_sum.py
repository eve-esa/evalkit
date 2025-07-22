import json
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Dict, List

import yaml
from litellm import OpenAI

PROMPT_PATH = Path(__file__).parent / "../metrics/llm_judge/prompts/summary_eval.yaml"
JUDGE_MODEL = "mistral-large-latest"
MAX_PARALLEL_REQUESTS = 10
_JUDGE_CLIENT = None
_PROMPT_TEMPLATE = None
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS)


def get_judge_client_and_prompt():
    global _JUDGE_CLIENT, _PROMPT_TEMPLATE

    if _JUDGE_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        _JUDGE_CLIENT = OpenAI(api_key=api_key)
        print(
            f"Initialized LLMJ client with a pool of {MAX_PARALLEL_REQUESTS} workers."
        )

    if _PROMPT_TEMPLATE is None:
        config = yaml.safe_load(PROMPT_PATH.read_text())
        _PROMPT_TEMPLATE = config["prompt"]

    return _JUDGE_CLIENT, _PROMPT_TEMPLATE


def judge_with_llm(sample: Dict) -> Dict:
    client, prompt_template = get_judge_client_and_prompt()

    json_schema = {
        "type": "object",
        "properties": {
            "relevance_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "coherence_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "factuality_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "conciseness_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "overall_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        },
        "required": [
            "relevance_score",
            "coherence_score",
            "factuality_score",
            "conciseness_score",
            "overall_score",
        ],
    }

    # Default response if the judge fails
    default_error_response = {
        "relevance_score": 1,
        "coherence_score": 1,
        "factuality_score": 1,
        "conciseness_score": 1,
        "overall_score": 1,
    }

    format_instructions = (
        "You must respond with a JSON object that strictly follows this schema:\n"
        f"{json.dumps(json_schema, indent=2)}"
    )

    final_prompt = prompt_template.format(
        document=sample["document"],
        prediction=sample["prediction"],
        reference=sample["reference"],
        format_instructions=format_instructions,
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=250,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM response content is None")
        data = json.loads(content)
        if "overall_score" in data:
            return data
        else:
            print(f"Warning: Judge response missed required keys. Data: {data}")
            return default_error_response
    except Exception as e:
        print(f"Error during LLM judge call: {e}. Defaulting all scores to 1.")
        return default_error_response


def process_results(doc: dict, results: list[str]) -> dict:
    """Submits a judging job and returns a future that will resolve to the evaluation
    dictionary."""
    sample = {
        "document": doc["input"],
        "prediction": results[0],
        "reference": doc["output"],
    }
    main_future = _EXECUTOR.submit(judge_with_llm, sample)

    return {
        "relevance": _EXECUTOR.submit(
            lambda f: f.result().get("relevance_score", 1), main_future
        ),
        "coherence": _EXECUTOR.submit(
            lambda f: f.result().get("coherence_score", 1), main_future
        ),
        "factuality": _EXECUTOR.submit(
            lambda f: f.result().get("factuality_score", 1), main_future
        ),
        "conciseness": _EXECUTOR.submit(
            lambda f: f.result().get("conciseness_score", 1), main_future
        ),
        "llm_judge": _EXECUTOR.submit(
            lambda f: f.result().get("overall_score", 1), main_future
        ),
    }


def aggregate_mean_score(items: List[Future[int]]) -> float:
    resolved_scores = [future.result() for future in items]
    return mean(resolved_scores) if items else 0.0
