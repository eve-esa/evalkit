import datasets
import re
import evaluate
from openai import OpenAI
import time
import os
import logging
from ranker import GPTPreferenceRanker

logging.getLogger("openai").setLevel(logging.ERROR)


with open('prompt', 'r') as f:
    prompt = f.read()

def create_context_prompt(doc) -> str:
    context = doc['context']
    question = doc['question']

    formatted_prompt = prompt.format(context, question)

    return formatted_prompt


def process_answer(answer):
    # Try JSON-style answers first
    json_answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', answer)
    if json_answer_match:
        return json_answer_match.group(1).strip()

    # Try plain format like: "Answer: ..."
    plain_answer_match = re.search(r'Answer\s*:\s*(.*)', answer)
    if plain_answer_match:
        return plain_answer_match.group(1).strip()

    # Otherwise, fallback to last non-empty line (excluding the prompt)
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return ""


def process_results(doc, results):
    model_pred = results[0]
    ranker = GPTPreferenceRanker()
    result = ranker.evaluate(model_pred, doc['answer'], doc['original_answer'])

    if result != 1:
        result = 0

    return {'win': result}


