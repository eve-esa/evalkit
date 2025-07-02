from metrics.metrics import bertscore_indus, rouge, cosine_sim
from metrics.llm_judge.correctness import LLMCorrectnessEvaluator

prompt_path = 'metrics/llm_judge/prompts/summary_eval.yaml'

def process_results(doc, results):
    reference = [doc['output']]
    cosine_score = cosine_sim(reference, results)
    bertscore_score = bertscore_indus(reference, results)

    judge = LLMCorrectnessEvaluator(prompt_path=prompt_path)

    sample = {'prediction': results[0], 'reference': doc['output']}

    judge_score = judge.judge(sample)

    scores = {
        'cosine_sim': cosine_score,
        'llm_judge': judge_score
    }
    #scores.update(rouge_score)
    scores.update(bertscore_score)

    return scores

