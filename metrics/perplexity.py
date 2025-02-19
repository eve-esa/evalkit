import json
import os
import evaluate
import pandas as pd
import click

def compute_perplexity(model_path, predictions, batch_size):
    # Try to load the metric
    try:
        ppl = evaluate.load('perplexity')
    except FileNotFoundError:
        # If it is not found clone the repo and load the metric
        os.system('git clone https://github.com/huggingface/evaluate.git')
        ppl = evaluate.load('evaluate/metrics/perplexity/perplexity.py')

    # Compute the perplexity
    perplexity = ppl.compute(predictions=predictions, model_id=model_path, batch_size=batch_size)

    return perplexity


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--dataset_path', help='Path to the dataset in jsonl format')
@click.option('--batch_size', default=8, help='Batch size for evaluation')
@click.option('--output_path', default='perplexity.json', help='Path to save the perplexity value')
def main(model_path, dataset_path, output_path='perplexity.json', batch_size=8):
    print(f'Evaluating perplexity on model {model_path} and dataset {dataset_path}...')

    # Assert it is a jsonl file
    assert dataset_path.endswith('.jsonl'), "Dataset must be in jsonl format"
    

    # Load the dataset
    df = pd.read_json(dataset_path, lines=True)

    # Assert there is a text column
    assert 'text' in df.columns, "Dataset must have a 'text' column"

    texts = df['text'].tolist()

    # Compute perplexity
    perplexity_value = compute_perplexity(model_path, texts, batch_size)

    # Save the perplexity value
    with open(output_path, 'w') as f:
        json.dump({'perplexity': perplexity_value}, f, indent=4)


if __name__ == '__main__':
    main()
