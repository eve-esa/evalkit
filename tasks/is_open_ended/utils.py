from metrics.metrics import bertscore_indus, rouge, cosine_sim
from metrics.llm_judge.correctness import LLMCorrectnessEvaluator

prompt_path = 'metrics/llm_judge/prompts/qa_eval.yaml'

def process_results(doc, results):
    reference = [doc['answer']]
    rouge_score = rouge(reference, results)
    cosine_score = cosine_sim(reference, results)
    bertscore_score = bertscore_indus(reference, results)

    judge = LLMCorrectnessEvaluator(prompt_path=prompt_path, results_file='test.txt')

    sample = {'question': doc['question'], 'output': results[0], 'reference': doc['answer']}

    judge_score = judge.judge(sample)

    scores = {
        'cosine_sim': cosine_score,
        'llm_judge': judge_score
    }
    scores.update(rouge_score)
    scores.update(bertscore_score)

    return scores

