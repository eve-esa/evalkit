# Evalkit

This repository contains the evaluation tasks and metrics for evaluating models on Earth Observation specific benchmarks.

Evalkit is built on top of the [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), a unified framework for testing generative language models on a large number of different evaluation tasks.

For more information refer to the complete guide of the project: [EVE Guide](eve-esa.github.io/eve-guide/).

## Earth Virtual Expert (EVE)

**Earth Virtual Expert (EVE)** aims to advance the use of Large Language Models (LLMs) within the Earth Observation (EO) and Earth Science (ES) community.

- Website: https://eve.philab.esa.int/  
- HuggingFace: https://huggingface.co/eve-esa
- Other repositories: https://github.com/eve-esa
  
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
source .venv/bin/activate  
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

## Usage

### Running Evaluations with Configuration File

The recommended way to run evaluations is using the YAML configuration file `scripts/config/evals.yaml`. This allows you to configure multiple models, tasks, and evaluation parameters in one place.

#### Configuration File Structure

The `evals.yaml` file has the following structure:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: your-judge-base-url
  judge_name: your-judge-name
  tasks:
    - name: mcqa_single_answer
      num_fewshot: 0
      max_tokens: 10000
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


## EO Tasks

The EVE-evaluation framework includes the following Earth Observation-specific evaluation tasks:

### Core Tasks

1. **mcqa_single_answer** - Multiple choice questions with a single correct answer
   - Dataset: `eve-esa/mcqa-single-answer`
   - Metrics: Accuracy
   - Evaluates model's ability to select the correct answer from multiple options

2. **mcqa_multiple_answer** - Multiple choice questions with multiple correct answers
   - Dataset: `eve-esa/mcqa-multiple-answers`
   - Metrics: Intersection over Union (IoU), Accuracy
   - Tests model's ability to identify all correct answers among options

3. **open_ended** - Open-ended question answering
   - Dataset: `eve-esa/open-ended`
   - Metrics: LLM-as-judge
   - Tests model's ability to generate comprehensive answers to EO questions

4. **open_ended_w_context** - Open-ended QA with context documents
   - Dataset: `eve-esa/open-ended-w-context`
   - Metrics: LLM-as-judge
   - Evaluates RAG-style question answering using provided context

5. **hallucination_detection** - Detection of hallucinations in EO answers
   - Dataset: `eve-esa/hallucination-detection`
   - Metrics: Accuracy, Precision, Recall, F1
   - Tests model's ability to identify false or unsupported information

6. **refusal** - Appropriate refusal when context is insufficient
   - Dataset: `eve-esa/refusal`
   - Metrics: LLM-as-judge
   - Evaluates model's ability to refuse answering when information is unavailable

All tasks support few-shot evaluation and can be configured with custom parameters in the `evals.yaml` configuration file.

## Development

### Adding New Tasks

To add a new evaluation task:

1. Create a new directory in `tasks/` with your task name
2. Add a YAML configuration file defining the task
3. Implement any custom metrics in `metrics/`
4. Add the task to your `evals.yaml` configuration

### Custom Metrics

Custom metrics are implemented in the `metrics/` package. See existing metrics for examples of how to implement new evaluation metrics.

## Funding

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.

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

## License

This project is released under the Apache 2.0 License - see the [LICENSE](LICENSE) file for more details.




