from perplexity import compute
from datasets import load_dataset
import click
import json


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--output_path', default='perplexity.json', help='Path to save the perplexity value')
@click.option('--split', default='validation', help='Dataset split to evaluate on')
def main(model_path, output_path='perplexity.json', split='validation'):
    # Load Mini Pile
    dataset = load_dataset("JeanKaddour/minipile", split=split)

    # Compute perplexity
    perplexity = compute(dataset['text'], model_id=model_path)

    # Save the perplexity value
    with open(output_path, 'w') as f:
        json.dump(perplexity, f, indent=4)


if __name__ == '__main__':
    main()