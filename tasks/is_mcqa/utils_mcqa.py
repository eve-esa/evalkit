import datasets


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
    return {"output": ", ".join(row["answers"])}


def doc_to_text(doc):
    string = f"{doc['question']}\n"

    for label, txt in zip(doc["choices"]["label"], doc["choices"]["text"]):
        string += f"{label}. {txt}\n"

    return string


def process_answer(answer):
    answers_list = answer.split(",")
    answers_list = [ans.strip() for ans in answers_list]
    return answers_list


def process_dataset(dataset: datasets.Dataset):
    return dataset.map(map_to_answers)


def process_results(doc: datasets.Dataset, results):
    preds = results[0]
    references = doc["answers"]
    preds = process_answer(preds)
    subset_acc = subset_accuracy(references, preds)
    jaccard = jaccard_index(references, preds)
    return {"acc": subset_acc, "IoU": jaccard}
