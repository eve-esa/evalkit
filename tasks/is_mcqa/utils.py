import datasets
import re

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
    return {'output': ', '.join(row['answers'])}


def doc_to_text(doc):
    string = f'{doc["question"]}\n'

    for label, txt in zip(doc['choices']['label'], doc['choices']['text']):
        string += f'{label}. {txt}\n'

    # if 'first' in doc.keys() and doc['first']:
    #     string = f'The following are multiple choice questions (with answers) about EO.\n' + string
    # if 'first' not in doc.keys():
    #     string = string + '\nAnswer: '

    return string

def get_answers(answer):
    answers_list = []
    matches = re.findall(r"(?:The answer is: |The answers are: )([A-Z](?:,\s*[A-Z])*)", answer)
    if matches:
        match = matches[0]
        # Split on spaces and strip
        answers_list = [ans.strip() for ans in match.split(",")]
    return answers_list

def process_answer(answer):
    # Split on commas
    answers_list = answer.split(",")
    # Strip each answer
    answers_list = [ans.strip() for ans in answers_list]
    return answers_list


def process_dataset(dataset: datasets.Dataset):
    return dataset.map(map_to_answers)

def process_results(doc: datasets.Dataset, results):
    #print(results)
    preds = results[0]
    references = doc["answers"]

    # Process preds
    preds = process_answer(preds)
    # Compute metrics
    subset_acc = subset_accuracy(references, preds)
    jaccard = jaccard_index(references, preds)

    # print('Question:', doc['question'])
    # print('Gold:', references)
    # print('Model out:', results[0])
    # print('Processed out:', preds)
    # print('Subset Acc:', subset_acc)
    # print('Jaccard:', jaccard)

    return {"acc": subset_acc, "IoU": jaccard}

