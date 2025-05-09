import datasets
import re
import evaluate
from openai import OpenAI
import time
import os
import logging

bertscore = evaluate.load("bertscore", lang="en-sci")
bleu = evaluate.load("bleu")
#bleurt = evaluate.load("bleurt", module_type="metric")
client = OpenAI()

# logger = logging.getLogger(__name__)
# Set the logging level to INFO
#logger.setLevel(logging.INFO)
# Set output file
logging.getLogger("openai").setLevel(logging.ERROR)
# Set file
#logging.basicConfig(filename='is_open_ended.log', level=logging.INFO)


def create_chat_prompt_2(question: str, llm_answer: str, answer: str) -> list[dict[str, str]]:
    """
    Prompt from https://arxiv.org/abs/2305.12421
    """
    sys_msg = """Here is a question, a set of golden answers (split with /), an AI-generated answer. Can you judge whether the AI-generated answer is correct according to the question and golden answers, simply answer Yes or No"""
    user_prompt = f"""QUESTION: {question}\n AI ANSWER: {llm_answer}\n GOLDEN ANSWER: {answer}\n"""

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt}
    ]


def create_chat_prompt(question: str, llm_answer: str, answer: str) -> list[dict[str, str]]:
    """
    Prompt from https://arxiv.org/html/2406.07545v1
    """
    sys_msg = """Evaluate the answer of a AI model to a question. You will be provided with the question, the AI model’s answer, and the correct answer. Your task is to evaluate the AI model’s response and determine whether it is Correct or Incorrect.
            Grade the AI model answers based ONLY on their factual accuracy. It is OK if the AI model answer contains more information than the true answer, as long as it does not contain any conflicting statements. Otherwise, it should be marked as Incorrect. Ignore differences in punctuation and phrasing between the AI model’s answer and the true answer.
            Example Format:
            QUESTION: question here
            AI ANSWER: AI answer here
            TRUE ANSWER: true answer here
            GRADE: Correct or Incorrect here
            Your response should include only the verdict without any justification or reasoning"""
    user_prompt = f"""QUESTION: {question}\n AI ANSWER: {llm_answer}\n TRUE ANSWER: {answer}\n GRADE: """
    #print(user_prompt)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_prompt}
    ]


def process_answer(answer):
    # Split on commas
    answer = answer.split("Q: ")[-1].strip()
    return answer


def get_chat_completion(prompt: list[dict[str, str]]) -> str:
    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(messages=prompt,
                                                        model='gpt-4o-mini-2024-07-18')
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            attempt += 1
            time.sleep(3)
    print(f"Failed to get chat completion after {max_retries} attempts")
    return "Cannot be answered"


def llm_as_judge(question: str, llm_answer: str, answer: str) -> dict[str, int]:
    prompt = create_chat_prompt(question, llm_answer, answer)
    llm_judge = get_chat_completion(prompt)
    # Parse Yes/No into 1, 0 for accuracy
    llm_judge = 0 if 'incorrect' in llm_judge.lower() else 1

    # SECOND PROMPT
    #prompt2 = create_chat_prompt_2(question, llm_answer, answer)
    #llm_judge2 = get_chat_completion(prompt2)
    # Parse Yes/No into 1, 0 for accuracy
    #if 'yes' in llm_judge2.lower():
    #    llm_judge2 = 1
    #elif 'no' in llm_judge2.lower():
    #    llm_judge2 = 0
    #else:
    #    print('Answer not recognized:', llm_judge2)

    return {"llm_judge_accuracy": llm_judge} #, "llm_judge_accuracy_2": llm_judge2}


def bertscore_metric(predictions: list[str], references: list[str], threshold=0.50) -> dict[str, float]:
    result = bertscore.compute(predictions=predictions, references=references, lang="en")
    f1 = result['f1'][0]
    precision = result['precision'][0]
    recall = result['recall'][0]
    accuracy = 1 if f1 > threshold else 0
    result = {'bertscore_f1': f1, 'bertscore_precision': precision, 'bertscore_recall': recall, 'bertscore_accuracy': accuracy}
    return result


def bleu_metric(predictions: list[str], references: list[list[str]], threshold=0.50) -> dict[str, float]:
    print(predictions)
    print(references)
    result = bleu.compute(predictions=predictions, refrences=references)
    bleu_score = result['bleu']
    accuracy = 1 if bleu_score > threshold else 0
    result = {'bleu_score': bleu_score, 'bleu_accuracy': accuracy}
    return result


#def bleurt_metric(predictions: list[str], references: list[str], threshold=0.50) -> dict[str, float]:
#    result = bleurt.compute(predictions=predictions, references=references)
#    score = result['scores'][0]
#    accuracy = 1 if score > threshold else 0
#    result = {'bleurt_score': score, 'bleurt_accuracy': accuracy}
#    return result


def process_results(doc, results):
    dict_results = {}
    #bleu_score = bleu_metric(predictions=results, references=[[doc['answer']]])
    bertscore_results = bertscore_metric(predictions=results, references=[doc['answer']])
    # bleurt_results = bleurt_metric(predictions=results, references=[doc['answer']])
    llm_result = llm_as_judge(doc['question'], results[0], doc['answer'])

    #dict_results.update(bleu_score)
    dict_results.update(bertscore_results)
    dict_results.update(llm_result)

    return dict_results


