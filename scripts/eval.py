import os
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import torch
import subprocess
import click


def parse_custom_csv(csv_file, outfile=""):
    if not outfile:
        outfile = csv_file.replace(".csv", ".json")
    df = pd.read_csv(csv_file)
    df_filtered = df[['Question', 'Answer']]
    df_filtered.rename(columns={"Question": "question", "Answer": "answer"}, inplace=True)
    data_to_save = df_filtered.to_dict(orient='records')
    # add id
    data_to_save = [{**el, "id": str(idx)} for idx, el in enumerate(data_to_save)]

    with open(outfile, 'w') as f:
        json.dump(data_to_save, f, indent=4)

    custom_dataset = load_dataset("json", data_files=outfile, split="train")

    return custom_dataset


def load_checkpoint_model(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # local_path = f"{path}/model-out/{checkpoint_name}/"
    # local_path = f"{path}/{checkpoint_name}/"
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=os.environ["HF_TOKEN"],
                                              padding_side='left')

    return tokenizer, model


def gen_preds(examples, generator, max_length=150):
    preds = []
    refs = examples['answer']

    # TODO - inference using pipeline is slow, it would be better to use batching or some inference framework as Vllm
    for sample in tqdm(examples):
        question = sample["question"]
        generated_answer = \
            generator(question, max_length=max_length, do_sample=True, truncation=True, return_full_text=False)[0][
                "generated_text"].strip()

        preds.append(generated_answer)
    return preds, refs


def mmlu_eval(model_path, output_path=None, file_name='mmlu.json'):
    if output_path is None:
        output_path = os.path.join(model_path, 'evaluation')

    #out = subprocess.run(
    #    ["lm_eval", "--model hf", "--model_args pretrained=" + model_path, "--tasks mmlu", "--device cuda:0",
    #     "--batch_size 8", "--output_path " + output_path], capture_output=True, text=True, shell=True)
    os.system(
        f"lm_eval --model hf --model_args pretrained={model_path} --tasks mmlu --device cuda:0 --batch_size 8 --output_path {output_path}")

    model_name = model_path.split('/')[-1]
    # Workaround to get the output file since lm_eval ignores the output_path argument
    tmp_dir = [x for x in os.listdir(output_path) if model_name in x][0]
    result_file = [x for x in os.listdir(os.path.join(output_path, tmp_dir)) if 'results' in x][0]

    os.system(f"mv {os.path.join(output_path, tmp_dir, result_file)} {os.path.join(output_path, file_name)}")
    os.system(f"rm -r {os.path.join(output_path, tmp_dir)}")
    # print(out.stdout)
    # print(out.stderr)


def download_qa_dataset():
    subprocess.run(['wget', '--no-check-certificate',
                    'https://docs.google.com/uc?export=download&id=1TbzMQ1wmWlKsf7QBSFJJYV4iKct49vVY', '-O',
                    '50_Q&A_Pairs.csv'])


def bertscores_eval(generator):
    qa_dataset = '50_Q&A_Pairs.csv'
    if qa_dataset not in os.listdir():
        download_qa_dataset()

    # build custom QA set
    custom_dataset = parse_custom_csv("50_Q&A_Pairs.csv")


    # generate preds on custom EO QA set
    preds, refs = gen_preds(custom_dataset, generator)

    # evaluate bertscore on custom open-ended QA testset
    metric = evaluate.load("bertscore")
    final_score = metric.compute(predictions=preds, references=refs, lang="en")

    final_score['f1_mean'] = sum(final_score['f1']) / len(final_score['f1'])
    final_score['precision_mean'] = sum(final_score['precision']) / len(final_score['precision'])
    final_score['recall_mean'] = sum(final_score['recall']) / len(final_score['recall'])

    return final_score


def evaluate_model(model_path: str, metric='all'):
    # Load environment variables
    # dotenv.load_dotenv()
    print(f"Evaluating {model_path}...")

    if metric == 'all' or metric == 'mmlu':
        print('Evaluation on MMLU...')

        if os.path.exists(os.path.join(model_path, 'evaluation', 'mmlu.json')):
            print('MMLU evaluation already exists, skipping...')
        else:
            mmlu_eval(model_path)


    if metric == 'all' or metric == 'qa50':
        print('Evaluation on QA50.')
        if os.path.exists(os.path.join(model_path, 'evaluation', 'qa50.json')):
            print('QA50 evaluation already exists, skipping...')
        else:
            # build model per checkpoint
            tokenizer, model = load_checkpoint_model(model_path)

            generator = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
            )

            # evaluate on custom QA dataset
            bertscores = bertscores_eval(generator)
            output = os.path.join(model_path, 'evaluation')
            os.makedirs(output, exist_ok=True)
            output = os.path.join(output, 'qa50.json')



            with open(output, 'w') as f:
                json.dump(bertscores, f, indent=4)


def eval_all_checkpoints(model_path: str):
    checkpoints = get_checkpoints_path(model_path)
    for checkpoint in checkpoints:
        evaluate_model(checkpoint)


def get_checkpoints_path(model_path: str):
    checkpoints_number = [int(f.name.split('-')[-1]) for f in os.scandir(model_path) if
                          f.is_dir() and 'checkpoint' in f.name]
    # sort checkpoints
    checkpoints_number.sort()
    checkpoints = [f"{model_path}/checkpoint-{checkpoint}" for checkpoint in checkpoints_number]
    return checkpoints

@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--metric', help='Metric to evaluate', default='all')
@click.option('--run_folder', help='Run evaluation on all checkpoints', is_flag=True)
def main(model_path, run_folder, metric='all'):
    print(model_path)
    if run_folder:
        eval_all_checkpoints(model_path)
    else:
        evaluate_model(model_path, metric)


if __name__ == '__main__':
    # Check if HF_TOKEN is set
    if 'HF_TOKEN' not in os.environ:
        raise ValueError("Please set the HF_TOKEN environment variable: HF_TOKE=<your_token>")
    main()
