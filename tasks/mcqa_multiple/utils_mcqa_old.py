import sys
from pathlib import Path

import datasets

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels


def subset_accuracy(references, predictions):
    correct_count = 0
    for correct, pred in zip(references, predictions):
        if set(correct) == set(pred):
            correct_count += 1
    return correct_count / len(references)


def jaccard_index(references, predictions):
    jaccard_scores = []
    for correct, pred in zip(references, predictions):
        intersection = len(set(correct) & set(pred))
        union = len(set(correct) | set(pred))
        jaccard_scores.append(intersection / union)

    if len(jaccard_scores) == 0:
        return 0
    return sum(jaccard_scores) / len(jaccard_scores)


def map_to_answers(row):
    return {"output": ", ".join(row["Answers"])}


def doc_to_text(doc):
    """
    Format document into MCQA prompt.

    Expected doc format:
    - 'Question': The question text
    - 'Choices': Dict with 'label' and 'text' lists
    """
    string = f"{doc['Question']}\n"

    # Choices is a dict with 'label' and 'text' lists
    choices = doc.get("Choices", {"label": [], "text": []})
    labels = choices.get("label", [])
    texts = choices.get("text", [])

    for label, txt in zip(labels, texts):
        string += f"{label}. {txt}\n"

    return string


def process_answer(answer):
    """
    Process model answer to extract letter choices.

    This is a wrapper around the common extract_labels function from mcqa_utils.

    Handles multiple formats:
    - Simple letters: "A", "B, C", "A,B,C"
    - Letter with period: "D.", "A. B."
    - Letter with full text: "D. Sea ice forms on ocean water..."
    - Multiple letters with text: "A. text, B. more text"
    - With prefix: "Answer: C. SIRAL"
    """
    return extract_labels(answer)


def process_dataset(dataset: datasets.Dataset):
    return dataset.map(map_to_answers)


def process_results(doc: datasets.Dataset, results):
    preds = results[0]
    references = doc["Answers"]
    preds = process_answer(preds)
    subset_acc = subset_accuracy(references, preds)
    jaccard = jaccard_index(references, preds)
    return {"acc": subset_acc, "IoU": jaccard}
