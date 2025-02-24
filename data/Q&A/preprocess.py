import datasets
import pandas as pd
import re


def extract_question_and_choices(text):
    if not isinstance(text, str):  # Ensure valid input
        return text, {}

    # Regex to match any labeled option (e.g., "A)", "1)", "III)", "D1)")
    pattern = r"(?m)^\s*(\w+)\)\s*(.+)"

    # Find all matches for options
    matches = re.findall(pattern, text)

    choices_dict = {'label': [], 'text': []}
    # Convert to dictionary
    for label, choice in matches:
        choices_dict['label'].append(label)
        choices_dict['text'].append(choice)

    # Remove options from text to keep only the question
    question_text = re.split(pattern, text, maxsplit=1)[0].strip()

    return question_text, choices_dict


def strip_blank_lines(text):
    '''Remove all the double newlines  and extra spaces'''
    if pd.isna(text):
        return text
    return "\n".join([line.strip() for line in text.strip().split("\n") if line.strip()])


def create_alpaca_entry(row):
    """Create an Alpaca format entry for the given content."""
    question = row['question']
    choices = row['choices']
    answers = row['answers']

    # Create the input prompt
    options = "\n".join([f"{label}) {text}" for label, text in zip(choices['label'], choices['text'])])
    input = f"{question}\n\n{options}"
    output = ("The answer is: " if len(answers) == 1 else "The answers are: ") + ', '.join(answers)

    instruction = ("Answer the given multiple choice question. There may be more than one "
                   "correct option")

    return {
        "instruction": instruction,
        "input": input,
        "output": output
    }


def add_mcq_options(text):
    if pd.isna(text):  # Handle missing values
        return text

    # Split the text into lines
    lines = text.strip().split("\n")

    # Ensure there are at least two lines (question + options)
    if len(lines) < 2:
        return text

    question = lines[0]  # First line is the question
    options = lines[1:]  # Remaining lines are answer choices

    # Format the answer choices with A), B), C), D)...
    formatted_options = [f"{chr(65 + i)}) {opt.strip()}" for i, opt in enumerate(options)]

    return f"{question}\n\n" + "\n".join(formatted_options)


def remove_space(answer):
    if pd.isna(answer):
        return answer
    return answer.replace(' ', '')


def to_hf_dataset(df, output_path):
    hf_dataset = datasets.Dataset.from_pandas(df)
    #hf_dataset.push_to_hub("antoniolopez00/MOOCQAs")
    return hf_dataset


def to_alpaca(data):
    data = data.map(create_alpaca_entry, remove_columns=['question', 'choices', 'answers', '__index_level_0__'])
    data.to_json("MOOCQAs.jsonl", lines=True)
    return data


def process():
    df = pd.read_csv('MOOCQAs.csv')

    # Remove rows with missing values
    df = df.dropna()

    df['Question'] = df['Question'].apply(strip_blank_lines).apply(add_mcq_options)
    # Save an intermerdiate format for sanity check
    df.to_csv('MOOCQAs_formatted.csv', index=False)
    df[["question", "choices"]] = df["Question"].apply(lambda x: pd.Series(extract_question_and_choices(x)))

    # Make answer a list of answers
    df["answers"] = df["Answer"].apply(lambda x: [ans.strip() for ans in x.split(",")])

    # Drop the original 'answer' column
    df = df.drop(columns=["Answer", 'Question'])

    # Add identifier column
    df.reset_index(drop=True)

    df.columns = df.columns.str.lower()

    hf_dataset = to_hf_dataset(df, "MOOCQAs")
    alpaca = to_alpaca(hf_dataset)

    return hf_dataset


def main():
    process()


if __name__ == '__name__':
    main()
