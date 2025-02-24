import json
import os
import pandas as pd
import click
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

import evaluate
from evaluate import logging
from tqdm import tqdm

import json
import evaluate
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import click
from tqdm import tqdm


# Create a custom Dataset class to handle batching
# Custom dataset using batch tokenization
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Function to compute perplexity

def compute2(predictions, model_id, batch_size: int = 1):

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    device = model.device

    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return encodings

    dataset = TextDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    ppls_list = []

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")

    stride = 1
    max_length = 2048
    with torch.no_grad():
        for batch in tqdm(dataloader):
            prev_end_loc = 0
            encoding = batch['input_ids']
            attn_mask = batch['attention_mask']
            seq_len = encoding.size(1)

            perplexity_sum = torch.zeros(encoding.size(0), device=device)
            perplexity_count = torch.zeros(encoding.size(0), device=device)
            for begin_loc in tqdm(range(0, seq_len, stride)):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encoding[:, begin_loc:end_loc].to(device)
                attn_mask = attn_mask.to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    logits = outputs.logits

                # Shift for skipping the last token (since does not have a target)
                shift_logits = logits[..., :-1, :].contiguous()
                # Shift for skipping the first token (since does not have a label that could be predicted)
                shift_labels = target_ids[..., 1:].contiguous()
                # Shift for skipping the first token
                shift_attention_mask_batch = attn_mask[..., 1:].contiguous()


                print(shift_logits.shape)
                print(shift_attention_mask_batch.shape)

                perplexity_sum += (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                perplexity_count += shift_attention_mask_batch.sum(1)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
            # avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
            # ppl = torch.exp(avg_nll)
            ppls_list.extend(torch.exp(perplexity_sum / perplexity_count).tolist())

    return {"mean_perplexity": torch.mean(torch.tensor(ppls_list)).item(), 'perplexities': ppls_list}


def compute(predictions: list[str], model_id: str, batch_size: int = 1):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    device = model.device

    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return encodings

    dataset = TextDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    ppls_list = []
    ppls = 0
    count = 0

    stride = 128
    max_length = 2048
    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            prev_end_loc = 0
            encoding = batch['input_ids']
            seq_len = encoding.size(1)
            for begin_loc in tqdm(range(0, seq_len, stride)):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
                input_ids = encoding[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100


                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                # Accumulate the total negative log-likelihood and the total number of tokens
                num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
                batch_size = target_ids.size(0)
                num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens
                # The main difference is where the mean is made


                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)

            # # Compute perplexity for the current batch
            # ppl = torch.exp(avg_nll)
            # # Add it to the list of perplexities
            # ppls_list.append(ppl.item())
            # # Update the total perplexity and the total number of examples
            # ppls += ppl.item()
            # count += batch_size

    return {"mean_perplexity": ppl.item()}


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--dataset_path', help='Path to the dataset in jsonl format')
@click.option('--output_path', default='perplexity.json', help='Path to save the perplexity value')
@click.option('--batch_size', default=1, help='Batch size for tokenization')
def main(model_path, dataset_path, output_path='perplexity.json', batch_size=1):
    print(f'Evaluating perplexity on model {model_path} and dataset {dataset_path}...')

    # Assert it is a jsonl file
    assert dataset_path.endswith('.jsonl'), "Dataset must be in jsonl format"

    # Load the dataset
    df = pd.read_json(dataset_path, lines=True)

    # Assert there is a text column
    assert 'text' in df.columns, "Dataset must have a 'text' column"

    texts = df['text'].tolist()

    # Compute perplexity
    metric = compute2(texts, model_path, batch_size=batch_size)

    # Save the perplexity value
    with open(output_path, 'w') as f:
        json.dump(metric, f, indent=4)


if __name__ == '__main__':
    main()
