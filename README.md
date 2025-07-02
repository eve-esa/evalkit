# EVE-evaluation
This repository contains the evaluation tasks and metrics for evaluate the model on Earth Observation specific benchmarks.

# Set-up
Clone the repo
```bash
git clone -b dev https://github.com/eve-esa/eve-evaluation.git
````

Copy models weights from s3 (optional if you want to perform local evaluation)
```bash
aws s3 cp s3://llm4eo-s3/eve_checkpoint_data/{your_folder} . --recursive --exclude "*.pt"
```

# Init environment
Init local env
```bash
python -m venv venv
source venv/bin/activate
```

Install requirements
```bash
pip install -r requirements.txt
```
Run setup script
```bash
source ./setup.sh
```


# Earth Observation tasks

## Imperative Space MCQA
Multiple-Choice question answer (432 samples) with arbitrary number of options and arbitrary number of correct options ([link](https://huggingface.co/datasets/eve-esa/eve-is-mcqa)).

The evaluation metrics are the following:
- Intersection Over Union: to evaluate also partially correct answers
- Accuracy: exact match between the set of correct answer and predicted


Run evaluation (example with GPT)
``` bash
lm_eval --model openai-completions --model_args model=gpt-4o-mini-2024-07-18 --tasks is_mcqa --include tasks
```


## Imperative Space Open-Ended
Open-ended question answer pairs (313) from MOOC exams ([link](https://huggingface.co/datasets/eve-esa/eve-is-open-ended)).

The evaluation metrics are the following:
- Cosine similarity between model answer and reference answer, the encoder model used is [Indus](https://huggingface.co/nasa-impact/nasa-smd-ibm-st-v2) a fine-tuned encode on scientific articles
- LLM-as-judge: GPT4o is prompted to evaluate if the model answer is correct. The model is prompted with: question, model answer and reference answer.
- BERTScore using Indus as encoder model.


Run evaluation (example with GPT)
``` bash
lm_eval --model openai-completions --model_args model=gpt-4o-mini-2024-07-18  --tasks is_open_ended --include tasks
```

## Imperative Space Open-Ended Hard

Subset of manually identified hard question from the Imperative Space Open-Ended ([link](https://huggingface.co/datasets/eve-esa/hardest-50-qna)). 

The evaluation metrics are the following:
- Cosine similarity between the abstract and the generated summary, the encoder model used is [Indus](https://huggingface.co/nasa-impact/nasa-smd-ibm-st-v2) a fine-tuned encode on scientific articles.
- LLM-as-judge: GPT4o is prompted to evaluate if the model summary is correct. The model is prompted with: the original abstract and the generated summary.
- BERTScore using Indus as encoder model.
- Preference: GPT4o is prompted to select the best answer between the one generated and a comparative model, in this case Llama-8B ((link)[https://huggingface.co/datasets/eve-esa/llama-rag]). The win rate is took as metric.

Run evaluation (example with GPT)

``` bash
lm_eval --model openai-completions --model_args model=gpt-4o-mini-2024-07-18  --tasks is_open_ended_hard --include tasks
```

## Earth Observation Summarization

Summarization dataset generated from a sample of scientific papers from our data (1k samples) ([link](https://huggingface.co/datasets/eve-esa/summarization_ds_10k_sample_split)). The summary is obtained by removing the abstract from the paper.

The evaluation metrics are the following:
- Cosine similarity between the abstract and the generated summary, the encoder model used is [Indus](https://huggingface.co/nasa-impact/nasa-smd-ibm-st-v2) a fine-tuned encode on scientific articles.
- LLM-as-judge: GPT4o is prompted to evaluate if the model summary is correct. The model is prompted with: the original abstract and the generated summary.
- BERTScore using Indus as encoder model.

Run evaluation (example with GPT)
``` bash
lm_eval --model openai-completions --model_args model=gpt-4o-mini-2024-07-18  --tasks eo_summarization --include tasks
```
