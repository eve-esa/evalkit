"""
Default utils for open-ended task.
"""

from metrics.judge_utils import (
    LoggableFuture,
    aggregate_llm_judge_default,
    get_executor,
    judge_qa_with_llm_default,
)


def process_results(doc: dict, results: list[str]) -> dict:
    """
    Auto-detects field names (question/answer vs Question/Answer).
    Returns dict with key 'llm_as_judge' to match task YAML metric name.
    """
    # Auto-detect field names
    if "Question" in doc:
        question_key = "Question"
        answer_key = "Answer"
    else:
        question_key = "question"
        answer_key = "answer"

    sample = {
        "question": doc[question_key],
        "output": results[0],
        "reference": doc[answer_key],
    }

    executor = get_executor()
    future = executor.submit(judge_qa_with_llm_default, sample)

    # Return with key 'llm_as_judge' to match task YAML metric name
    return {"llm_as_judge": LoggableFuture(future)}


def aggregate_llm_judge(results, **kwargs):
    """Aggregate LLM judge scores."""
    return aggregate_llm_judge_default(results, **kwargs)
