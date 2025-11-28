import re
import sys
from pathlib import Path

import datasets

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels


def doc_to_text(doc):
    """
    Format the document into a hallucination detection prompt.

    Expected doc format:
    - 'question': The original question
    - 'answer': The answer to evaluate for hallucination
    """
    question = doc.get('question', '')
    answer = doc.get('answer', '')

    string = f"Question: {question}\n"
    string += f"Answer: {answer}\n\n"
    string += "Is this answer hallucinated (contains false or unsupported information)?\n"
    string += "A. Yes\n"
    string += "B. No\n"

    return string


def process_answer(answer):
    """
    Process model answer to extract the choice (A or B).

    Returns a list containing the extracted label.
    """
    labels = extract_labels(answer)

    # If no valid label found, default to empty list
    if not labels:
        return []

    # Only take the first label if multiple are returned
    return [labels[0]] if labels else []


def map_label_to_letter(is_hallucinated):
    """
    Map boolean or string label to letter choice.

    Args:
        is_hallucinated: Can be bool, int, or string indicating if answer is hallucinated

    Returns:
        str: 'A' if hallucinated, 'B' if not
    """
    # Handle various formats
    if isinstance(is_hallucinated, bool):
        return 'A' if is_hallucinated else 'B'
    elif isinstance(is_hallucinated, int):
        return 'A' if is_hallucinated == 1 else 'B'
    elif isinstance(is_hallucinated, str):
        # Handle string representations
        lower = is_hallucinated.lower().strip()
        if lower in ['true', 'yes', '1', 'a', 'hallucinated']:
            return 'A'
        else:
            return 'B'
    else:
        return 'B'  # Default to not hallucinated


def calculate_f1(references, predictions):
    """
    Calculate F1 score for binary classification.

    Args:
        references: List of ground truth labels (letters)
        predictions: List of predicted labels (letters)

    Returns:
        float: F1 score
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for ref, pred in zip(references, predictions):
        ref_label = ref[0] if ref else 'B'
        pred_label = pred[0] if pred else 'B'

        # A = hallucinated (positive class)
        # B = not hallucinated (negative class)
        if ref_label == 'A' and pred_label == 'A':
            true_positives += 1
        elif ref_label == 'B' and pred_label == 'A':
            false_positives += 1
        elif ref_label == 'A' and pred_label == 'B':
            false_negatives += 1

    # Calculate F1
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        return 0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_accuracy(references, predictions):
    """
    Calculate accuracy for binary classification.
    """
    correct = 0
    for ref, pred in zip(references, predictions):
        ref_label = ref[0] if ref else 'B'
        pred_label = pred[0] if pred else 'B'
        if ref_label == pred_label:
            correct += 1

    return correct / len(references) if references else 0


def process_results(doc: datasets.Dataset, results):
    """
    Process results for hallucination detection task.

    Args:
        doc: Document containing ground truth label
        results: Model predictions

    Returns:
        dict: Metrics (accuracy and F1)
    """
    # Get model prediction
    pred_text = results[0]
    pred_labels = process_answer(pred_text)

    # Get ground truth label
    # The label in doc should indicate if the answer is hallucinated
    # Map it to letter format (A = hallucinated, B = not hallucinated)
    ground_truth_label = doc.get('label', doc.get('is_hallucinated', False))
    reference = [map_label_to_letter(ground_truth_label)]

    # Calculate metrics
    accuracy = calculate_accuracy([reference], [pred_labels])
    f1 = calculate_f1([reference], [pred_labels])

    return {
        "acc": accuracy,
        "f1": f1
    }