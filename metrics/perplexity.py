import json
import evaluate
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import click
from tqdm import tqdm

# Create a custom Dataset class to handle batching
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize each text and return a batch-friendly format
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return encoding


class Perplexity:
    """
    Class to compute perplexity of a model on a dataset.
    """

    def __init__(self, dataset: pd.DataFrame, model: AutoModelForCausalLM, tokenizer, batch_size=8):
        self.name = 'perplexity'
        # Assert that dataset contains the 'text' column
        assert 'text' in dataset.columns, "Dataset must contain 'text' column"
        self.model = model
        self.tokenizer = tokenizer
        # Pad token
        self.tokenizer.pad_token = tokenizer.eos_token

        self.dataset = TextDataset(dataset['text'], self.tokenizer)
        self.batch_size = batch_size
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)

    # Function to compute perplexity
    def __call__(self):
        total_log_likelihood = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                input_ids = batch['input_ids'].squeeze(1).to(self.model.device)

                # Compute the model's loss (log-likelihood)
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

                total_log_likelihood += loss * input_ids.size(1)  # Multiply by number of tokens in batch
                total_tokens += input_ids.size(1)  # Count number of tokens

        # Compute perplexity as exp of negative average log-likelihood
        perplexity = torch.exp(torch.tensor(total_log_likelihood / total_tokens)).item()
        return perplexity


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--dataset_path', help='Path to the dataset in jsonl format')
@click.option('--batch_size', default=8, help='Batch size for evaluation')
@click.option('--output_path', default='perplexity.json', help='Path to save the perplexity value')
def main(model_path, dataset_path, output_path='perplexity.json', batch_size=8):

    print(f'Evaluating perplexity on model {model_path} and dataset {dataset_path}...')
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Assert it is a jsonl file
    assert dataset_path.endswith('.jsonl'), "Dataset must be in jsonl format"

    # Load the dataset
    df = pd.read_json(dataset_path, lines=True)

    # Initialize the Perplexity class
    perplexity = Perplexity(df, model, tokenizer, batch_size=batch_size)

    # Compute perplexity
    perplexity_value = perplexity()

    # Save the perplexity value
    with open(output_path, 'w') as f:
        json.dump({'perplexity': perplexity_value}, f, indent=4)


if __name__ == '__main__':
    main()