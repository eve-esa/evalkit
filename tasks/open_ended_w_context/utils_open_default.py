import json
import os
import time
from concurrent.futures import Future
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

import yaml

from metrics.judge_utils import (
    JUDGE_FAILURE_SCORE,
    JUDGE_REQUEST_DELAY,
    LoggableFuture,
    get_executor,
    get_judge_client,
)

# Global state for prompt template
_QA_PROMPT_TEMPLATE = None


def get_qa_prompt_template() -> str:
    """Load and return the QA judge prompt template (length-neutral grounding)."""
    global _QA_PROMPT_TEMPLATE

    if _QA_PROMPT_TEMPLATE is None:
        prompt_path = (
            Path(__file__).parent.parent.parent / "metrics" / "prompts" / "llm_judge_qa_w_context_default.yaml"
        )
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
    Length-neutral document-grounding focused judge.

    Key features:
    - Does not penalize concise answers
    - Penalizes verbose formatting without substance
    - Judges on grounding accuracy

    Args:
        sample: Dictionary with "question", "output", "reference", and "documents" keys.
        model_name: Name of the judge model.
        api_key: API key for the judge.
        base_url: Base URL for the API.

    Returns:
        Dictionary with "score" (int 0-5) and "raw_output" (str) keys.
    """
    if model_name is None:
        model_name = os.getenv("JUDGE_NAME") or os.getenv("JUDGE_MODEL") or "mistral-medium-latest"

    client = get_judge_client(api_key=api_key, base_url=base_url)
    prompt_template = get_qa_prompt_template()

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
        documents=sample["documents"],
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
            print(
                f"Warning: Judge returned invalid score: {score}. Defaulting to {JUDGE_FAILURE_SCORE}."
            )
            return {"score": JUDGE_FAILURE_SCORE, "raw_output": response_content}

    except Exception as e:
        print(f"Error during LLM judge call: {e}. Defaulting to score {JUDGE_FAILURE_SCORE}.")
        return {"score": JUDGE_FAILURE_SCORE, "raw_output": f"Error: {str(e)}"}


def process_results(doc: dict, results: list[str]) -> dict:
    """
    Process results for open-ended questions with context.

    Extracts and passes the document context to the judge
    so it can verify grounding.
    """
    # Build the documents string from Doc 1, Doc 2, Doc 3
    documents_parts = []
    if doc.get("Doc 1"):
        documents_parts.append(f"**Document 1:**\n{doc['Doc 1']}")
    if doc.get("Doc 2"):
        documents_parts.append(f"**Document 2:**\n{doc['Doc 2']}")
    if doc.get("Doc 3"):
        documents_parts.append(f"**Document 3:**\n{doc['Doc 3']}")

    documents_str = "\n\n".join(documents_parts) if documents_parts else "(No documents provided)"

    # Truncate documents if too long (to avoid token limits)
    max_doc_chars = 8000  # Leave room for the rest of the prompt
    if len(documents_str) > max_doc_chars:
        documents_str = (
            documents_str[:max_doc_chars] + "\n\n[... documents truncated for length ...]"
        )

    sample = {
        "question": doc.get("Question", doc.get("question", "")),
        "output": results[0],
        "reference": doc.get("Answer", doc.get("answer", "")),
        "documents": documents_str,
    }

    executor = get_executor()
    future = executor.submit(judge_qa_with_llm, sample)

    return {"llm_as_judge_default": LoggableFuture(future)}


def doc_to_text(doc: dict) -> str:
    """Converts a document dictionary to a text representation."""
    context = ""
    idx = 1
    if doc.get("Doc 1"):
        context += f"Document {idx}: {doc['Doc 1']}\n"
        idx += 1
    if doc.get("Doc 2"):
        context += f"Document {idx}: {doc['Doc 2']}\n"
        idx += 1
    if doc.get("Doc 3"):
        context += f"Document {idx}: {doc['Doc 3']}\n"

    return f"Context: {context}\n\nQuestion: {doc['Question']}\n"


def aggregate_llm_judge(items) -> float:
    """Aggregate LLM judge results, normalized to 0-1 scale."""
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
