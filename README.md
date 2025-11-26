# EVE-evaluation

This repository contains the evaluation tasks and metrics for evaluating models on Earth Observation specific benchmarks.

EVE-evaluation is built on top of the [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), a unified framework for testing generative language models on a large number of different evaluation tasks.

## Installation

### 1. Clone the repository

```bash
git clone -b dev https://github.com/eve-esa/eve-evaluation.git
cd eve-evaluation
```

### 2. Set up Python environment

We recommend using Python 3.10 or higher. Install `uv` if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install the package

Install the package and all dependencies using uv:

```bash
uv sync
```

This will install:
- The EVE evaluation framework with all custom tasks and metrics
- The lm-evaluation-harness framework
- All required dependencies (PyTorch, transformers, datasets, etc.)

### 4. Run setup script

Execute the setup script to configure the environment:

```bash
source ./setup.sh
```

## Usage

### Running Evaluations with Configuration File

The recommended way to run evaluations is using the YAML configuration file `scripts/config/evals.yaml`. This allows you to configure multiple models, tasks, and evaluation parameters in one place.

#### Configuration File Structure

The `evals.yaml` file has the following structure:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411
  tasks:
    - name: is_mcqa
      num_fewshot: 0
      max_tokens: 10000
      judge_api_key: !ref judge_api_key
      judge_base_url: !ref judge_base_url
      judge_name: !ref judge_name
    - name: eo_summarization
      num_fewshot: 0
      max_tokens: 20000
      judge_api_key: !ref judge_api_key
      judge_base_url: !ref judge_base_url
      judge_name: !ref judge_name

wandb:
  enabled: true
  project: evaluations
  entity: your-wandb-entity
  run_name: my-evaluation-run
  api_key: your-wandb-api-key

models:
  - name: eve-esa/eve_v0.1
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 5
    timeout: 180
    tasks: !ref tasks

output_dir: evals_outputs
```

#### Configuration Options

**Constants Section:**
- Define reusable values that can be referenced throughout the config using `!ref`
- Common constants: API keys, base URLs, model names, task lists

**Tasks Configuration:**
- `name`: Task name (must match a task in the `tasks/` directory)
- `num_fewshot`: Number of few-shot examples (default: 0)
- `max_tokens`: Maximum tokens for model generation
- `limit`: Optional limit on number of samples to evaluate
- `judge_api_key`: API key for LLM-as-judge evaluation (if applicable)
- `judge_base_url`: Base URL for judge model API
- `judge_name`: Name/ID of judge model

**Models Configuration:**
- `name`: Model name/identifier
- `base_url`: API endpoint for the model
- `api_key`: Authentication key for the model API
- `temperature`: Sampling temperature (default: 0.0)
- `num_concurrent`: Number of concurrent API requests (default: 3)
- `timeout`: Request timeout in seconds (default: 300)
- `tasks`: List of tasks to run on this model (can reference `!ref tasks`)

**Weights & Biases (wandb) Configuration:**
- `enabled`: Enable/disable wandb logging
- `project`: Wandb project name
- `entity`: Wandb entity/organization
- `run_name`: Custom run name prefix (optional)
- `api_key`: Wandb API key

**Output Configuration:**
- `output_dir`: Directory where evaluation results will be saved

#### Running Evaluations

To run evaluations using the configuration file:

```bash
python scripts/evaluate.py scripts/config/evals.yaml
```

The script will:
1. Parse the configuration file
2. Run each task for each model sequentially
3. Save results to the specified output directory
4. Optionally log metrics and samples to Weights & Biases
5. Print a summary of all evaluations

#### Output Structure

Results are saved in the following structure:

```
{output_dir}/
  {model_name}/
    {task_name}/
      {model_name}/
        results_{timestamp}.json       # Evaluation metrics
        samples_{dataset}_{timestamp}.jsonl  # Individual samples
```

### Running Individual Tasks (Advanced)

You can also run individual tasks directly using the lm_eval command:

```bash
lm_eval --model openai-chat-completions \
        --model_args base_url=https://api.provider.com,model=model-name,num_concurrent=5 \
        --tasks {task_name} \
        --include tasks \
        --num_fewshot 0 \
        --output_path {output_dir} \
        --log_samples \
        --apply_chat_template
```

Set the API key as an environment variable:

```bash
export OPENAI_API_KEY=your-api-key
```

For tasks using LLM-as-judge metrics, also set:

```bash
export JUDGE_API_KEY=your-judge-api-key
export JUDGE_BASE_URL=https://api.provider.com/v1
export JUDGE_NAME=judge-model-name
```

## Earth Observation Tasks

### Imperative Space MCQA

Multiple-Choice question answer (432 samples) with arbitrary number of options and arbitrary number of correct options ([link](https://huggingface.co/datasets/eve-esa/eve-is-mcqa)).

**Evaluation metrics:**
- Intersection Over Union: to evaluate also partially correct answers
- Accuracy: exact match between the set of correct answer and predicted

**Task name:** `is_mcqa`

### Imperative Space Open-Ended

Open-ended question answer pairs (313) from MOOC exams ([link](https://huggingface.co/datasets/eve-esa/eve-is-open-ended)).

**Evaluation metrics:**
- Cosine similarity between model answer and reference answer, using [Indus](https://huggingface.co/nasa-impact/nasa-smd-ibm-st-v2) encoder
- LLM-as-judge: GPT4o evaluates if the model answer is correct
- BERTScore using Indus as encoder model

**Task name:** `is_open_ended`

### Imperative Space Open-Ended Hard

Subset of manually identified hard questions from the Imperative Space Open-Ended ([link](https://huggingface.co/datasets/eve-esa/hardest-50-qna)).

**Evaluation metrics:**
- Cosine similarity using [Indus](https://huggingface.co/nasa-impact/nasa-smd-ibm-st-v2) encoder
- LLM-as-judge: GPT4o evaluation
- BERTScore using Indus encoder
- Preference: GPT4o selects the best answer between generated and Llama-8B baseline

**Task name:** `is_open_ended_hard`

### Earth Observation Summarization

Summarization dataset generated from scientific papers (1k samples) ([link](https://huggingface.co/datasets/eve-esa/summarization_ds_10k_sample_split)).

**Evaluation metrics:**
- Cosine similarity using [Indus](https://huggingface.co/nasa-impact/nasa-smd-ibm-st-v2) encoder
- LLM-as-judge: GPT4o evaluation
- BERTScore using Indus encoder

**Task name:** `eo_summarization`

## Available Benchmarks

All available benchmarks and tasks can be found in the `tasks/` directory. Each task includes:
- Dataset loading configuration
- Prompt templates
- Evaluation metrics
- Task-specific parameters

## Development

### Adding New Tasks

To add a new evaluation task:

1. Create a new directory in `tasks/` with your task name
2. Add a YAML configuration file defining the task
3. Implement any custom metrics in `metrics/`
4. Add the task to your `evals.yaml` configuration

### Custom Metrics

Custom metrics are implemented in the `metrics/` package. See existing metrics for examples of how to implement new evaluation metrics.

## Support

For issues or questions, please open an issue on the GitHub repository.

## Citation

If you use this evaluation framework, please cite both EVE-evaluation and the underlying lm-eval harness:

```bibtex
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Abbasi, Baber and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  Le Noac'h, Alain and
                  Li, Haonan and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Ociepa, Chris and
                  Phang, Jason and
                  Reynolds, Laria and
                  Schoelkopf, Hailey and
                  Skowron, Aviya and
                  Sutawika, Lintang and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
