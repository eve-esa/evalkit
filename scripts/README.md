# README

## Overview

This folder contains two scripts for evaluation and plots generation.

The evaluation script supports the following benchmarks:

- **MMLU (Massive Multitask Language Understanding) Benchmark from **
- **Custom QA Dataset Evaluation using BERTScore**

It supports evaluation on single or multiple model checkpoints and integrates with Hugging Face's Transformers and the `lm_eval` framework.
## Requirements

Ensure to install the dependencies using the following command:

```bash
pip install -r requirements.txt
```
Be sure to have an HF_TOKEN variable in your enviroment. You can set it by running the following command:
```bash
export HF_TOKEN=<your_hf_token>
```

## Usage

### Running the Evaluation

To evaluate a specific model checkpoint:

```bash
python script.py --model_path /path/to/model --metric all
```

To evaluate all checkpoints in a given folder:

```bash
python script.py --model_path /path/to/run --run_folder
```

### Arguments

- `--model_path` (str): Path to the model checkpoint.
- `--metric` (str, default: `all`): Evaluation metric (`mmlu`, `qa50`, or `all`).
- `--run_folder` (flag): Run evaluation on all checkpoints in the given model path.

