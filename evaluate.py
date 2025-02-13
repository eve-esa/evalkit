import os
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import torch
import subprocess

torch.manual_seed(42)

MODEL_NAMES = ["warmup-run-256", "warmup-run-256-lr-85e-5", "warmup-run-256-lr-15e-4"]

def get_checkpoint_names(run_name, path):
  local_path = path #+ "/model-out/"
  dirs = [name for name in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, name))]
  checkpoints = [dir for dir in dirs if "checkpoint-" in dir]
  sorted_checpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
  return sorted_checpoints


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


def load_checkpoint_model(path, checkpoint_name):
  #local_path = f"{path}/model-out/{checkpoint_name}/"
  local_path = f"{path}/{checkpoint_name}/"
  model = AutoModelForCausalLM.from_pretrained(local_path, device_map='auto')
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", token=os.environ["HF_TOKEN"])

  return tokenizer, model


def load_vanilla_model(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  quantization_config = BitsAndBytesConfig(load_in_8bit=True)
  model = AutoModelForCausalLM.from_pretrained(model_name,
                                                quantization_config=quantization_config,
                                                device_map='auto')

  return tokenizer, model

def load_custom_hf_model(model_name, tokenizer="llama"):
  # model_repo = "eve-base-v0.1"
  # model_repo = "eve-instruct-v0.1"

  model_repo = f"eve-esa/{model_name}"

  if tokenizer == "llama":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
  quantization_config = BitsAndBytesConfig(load_in_8bit=True)
  model = AutoModelForCausalLM.from_pretrained(
      model_repo,
      device_map="auto",   # Uses available GPUs if any
      torch_dtype="auto",  # Automatically sets the correct tensor type
      use_safetensors=True, # Ensures it loads safetensors files
      quantization_config=quantization_config,
  )

  return tokenizer, model

def gen_preds(examples, generator):
  questions = []
  preds = []
  refs = []

  for sample in tqdm(examples):
      question = sample["question"]
      generated_answer = generator(question, max_length=150, do_sample=True, truncation=True)[0]["generated_text"].replace(question, "").strip()

      questions.append(question)
      preds.append({"id": sample["id"], "prediction_text": generated_answer, "no_answer_probability": 0.})
      refs.append({"id": sample["id"], "answers": {"answer_start": [0], "text": sample["answer"]}})

  return preds, refs

def eval_qa(preds, refs, metric="squad"):
  if metric=="squad":
    # Compute exact match and F1 scores
    metric = evaluate.load("squad")
    final_score = metric.compute(predictions=preds, references=refs)
  elif metric=="bertscore":
    metric = evaluate.load("bertscore")
    final_score = metric.compute(predictions=preds, references=refs, lang="en")

  return final_score



if __name__ == "__main__":
    for run_name in MODEL_NAMES:
      print(f"Running {run_name}...")

      local_path = f"./{run_name}"

      # download run with all checkpoints in drive
      #if not os.path.exists(local_path):
      #  download_run(run_name, local_path)

      # TEMP - REMOVE
      #continue

      # get checkpoint names
      checkpoints = get_checkpoint_names(run_name, local_path)

      # build model per checkpoint
      for checkpoint in checkpoints:
        tokenizer, model = load_checkpoint_model(local_path, checkpoint)

        generator = pipeline(
          task="text-generation",
          model=model,
          tokenizer=tokenizer,
          device_map="auto",
          pad_token_id=tokenizer.eos_token_id
          )

        # build custom QA set
        custom_dataset = parse_custom_csv("50_Q&A_Pairs.csv")

        # generate preds on custom EO QA set
        preds, refs = gen_preds(custom_dataset, generator)

        # evaluate bertscore on custom open-ended QA testset
        #bertscore = evaluate.load("bertscore")
        final_score = eval_qa(preds, refs, metric="bertscore")

        # output bertscore final_score to drive
        with open(f"{local_path}/{checkpoint}/evaluation/bertscore.json", "w") as f:
          json.dump(final_score, f)

        # generate MMLU preds
        output_mmlu = os.path.join(local_path, 'evaluation', 'mmlu.jsonl')
        local_model = os.path.join(local_path, checkpoint)
        #os.environ["LOCAL_MODEL"] = local_model
        #os.environ["OUTPUT_MMLU"] = output_mmlu

        subprocess.run(["lm_eval", "--model hf", f"--model_args pretrained={local_model}", "--tasks mmlu", "--device cuda:0", "--batch_size 8", f"--output_path {output_mmlu}"])


