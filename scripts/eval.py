import os
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
import click
import wandb



def load_checkpoint_model(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # local_path = f"{path}/model-out/{checkpoint_name}/"
    # local_path = f"{path}/{checkpoint_name}/"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=os.environ["HF_TOKEN"],
                                              padding_side='left')

    return tokenizer, model



def lm_eval(model_path, output_path=None, tasks=['mmlu']):
    if output_path is None:
        output_path = os.path.join(model_path, 'evaluation')
    file_name = ','.join(tasks) + '.json'

    tasks_str = ','.join(tasks)
    os.system(
        f"lm_eval --model vllm --model_args pretrained={model_path} --tasks {tasks_str} --output_path {output_path}")

    model_name = model_path.split('/')[-1]
    # Workaround to get the output file since lm_eval ignores the output_path argument
    print()
    tmp_dir = [x for x in os.listdir(output_path) if model_name in x][0]
    result_file = [x for x in os.listdir(os.path.join(output_path, tmp_dir)) if 'results' in x][0]

    os.system(f"mv {os.path.join(output_path, tmp_dir, result_file)} {os.path.join(output_path, file_name)}")
    os.system(f"rm -r {os.path.join(output_path, tmp_dir)}")



def load_on_wandb(step: int, metrics: dict, wandb_id: str, metric_prefix: str = 'eval_'):
    """
    Logs evaluation metrics to an existing Weights & Biases run.

    Args:
        step (int): The training or evaluation step number.
        metrics (dict): A dictionary of evaluation metrics to log.
        wandb_id (str): The unique ID of the wandb run.
    """
    wandb.init(id=wandb_id, resume="allow", reinit=True)

    # Add the prefix to the metrics
    metrics = {f"{metric_prefix}{k}": v for k, v in metrics.items()}

    # Log metrics at the specified step
    wandb.log(metrics, step=step)

    wandb.finish()


def evaluate_model(model_path: str, tasks=None):
    # Load environment variables
    # dotenv.load_dotenv()
    if tasks is None:
        tasks = ['mmlu']
    print(f"Evaluating {model_path}...")

    output = os.path.join(model_path, 'evaluation')
    os.makedirs(output, exist_ok=True)
    # Check if the metric is already computed
    file_name = ','.join(tasks) + '.json'
    #if os.path.exists(os.path.join(output, file_name)):
   #     print(f"Metrics already computed for {model_path}")
   # else:
    lm_eval(model_path, output, tasks=tasks)


def eval_all_checkpoints(model_path: str, metrics=None, wandb_id=None):
    if metrics is None:
        metrics = ['mmlu']
    checkpoints = get_checkpoints_path(model_path)
    for checkpoint in checkpoints:
        evaluate_model(checkpoint, metrics)


def get_checkpoints_path(model_path: str):
    checkpoints_number = [int(f.name.split('-')[-1]) for f in os.scandir(model_path) if
                          f.is_dir() and 'checkpoint' in f.name]
    # sort checkpoints
    checkpoints_number.sort()
    checkpoints = [f"{model_path}/checkpoint-{checkpoint}" for checkpoint in checkpoints_number]
    return checkpoints

@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--tasks', help='Metrics to evaluate', default='all')
@click.option('--run_folder', help='Run evaluation on all checkpoints', is_flag=True)
@click.option('--wandb_id', help='Id of the wandb run to upload to', default=None)
def main(model_path, run_folder, tasks='all', wandb_id=None):

    # Parse tasks by comma
    tasks = tasks.split(',')
    if run_folder:
        eval_all_checkpoints(model_path, tasks, wandb_id=wandb_id)
    else:
        evaluate_model(model_path, tasks)


if __name__ == '__main__':
    # Check if HF_TOKEN is set
    if 'HF_TOKEN' not in os.environ:
        raise ValueError("Please set the HF_TOKEN environment variable: HF_TOKE=<your_token>")
    main()
