from metrics.judge_utils import aggregate_llm_judge_default, process_qa_results_default


def process_results(doc, results, **kwargs):
    """Process results using default judge configuration."""
    return process_qa_results_default(doc, results, **kwargs)


def aggregate_llm_judge(results, **kwargs):
    """Aggregate LLM judge scores."""
    return aggregate_llm_judge_default(results, **kwargs)
