# Getting Started with Eve-evalkit

Eve-evalkit is built on top of the [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), which means it supports **all tasks available in the lm-evaluation-harness** in addition to the custom Earth Observation tasks.

## Quick Start

### 1. Installation

Follow the installation instructions in the [README](https://github.com/eve-esa/evalkit):

```bash
# Clone the repository
git clone https://github.com/eve-esa/evalkit.git
cd evalkit

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### 2. Running Evaluations

The recommended way to run evaluations is using the YAML configuration file. Create an `evals.yaml` file:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411
  tasks:
    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100
    - name: mcqa_single_answer
      num_fewshot: 2
      max_tokens: 1000

wandb:
  enabled: true
  project: eve-evaluations
  entity: your-wandb-entity
  run_name: my-evaluation
  api_key: your-wandb-api-key

models:
  - name: your-model-name
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 5
    timeout: 180
    tasks: !ref tasks

output_dir: evals_outputs
```

Run the evaluation:

```bash
python evaluate.py evals.yaml
```

---

## Configuration File Structure

### Constants Section

Define reusable values that can be referenced throughout the config using `!ref`:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411
  hf_token: your-huggingface-token  # Optional: for private datasets

  tasks:
    - name: task_name
      num_fewshot: 0
      max_tokens: 1000
      judge_api_key: !ref judge_api_key  # Reference to constant
      judge_base_url: !ref judge_base_url
      judge_name: !ref judge_name
```

### Tasks Configuration

Each task can have the following parameters:

```yaml
tasks:
  - name: task_name              # Required: Task identifier
    task_name: base_task         # Optional: Base task name (for custom naming)
    num_fewshot: 0               # Number of few-shot examples (default: 0)
    max_tokens: 1000             # Maximum tokens for generation (default: 512)
    temperature: 0.0             # Sampling temperature (default: 0.0)
    limit: 100                   # Optional: Limit number of samples to evaluate
    seed: 1234                   # Optional: Random seed for reproducibility
    model_type: local-chat-completions  # Optional: Model type (local-chat-completions, eve-api)
    judge_api_key: api-key       # Required for single-judge LLM-as-judge tasks
    judge_base_url: base-url     # Required for single-judge LLM-as-judge tasks
    judge_name: model-name       # Required for single-judge LLM-as-judge tasks
    judges: []                   # Optional: List of multiple judges for multi-judge evaluation
```

#### Random Seed for Reproducibility

The `seed` parameter allows you to control randomness in evaluations for reproducible results:

```yaml
tasks:
  - name: open_ended_0_shot_seed_1234
    task_name: open_ended
    num_fewshot: 0
    max_tokens: 10000
    seed: 1234  # Fixed seed ensures same results across runs
```

**Benefits of using seeds:**
- Reproduce exact same results across multiple runs
- Debug evaluation issues with consistent behavior
- Compare model performance with controlled randomness
- Run variance analysis by using multiple different seeds

**Example: Running same task with different seeds**

```yaml
tasks:
  - name: open_ended_seed_1234
    task_name: open_ended
    seed: 1234

  - name: open_ended_seed_5678
    task_name: open_ended
    seed: 5678

  - name: open_ended_seed_9012
    task_name: open_ended
    seed: 9012
```

This configuration runs the same task three times with different random seeds to analyze variance in results.

#### Multi-Judge Evaluation

For more robust evaluation, you can use multiple judges to evaluate each answer. This is particularly useful for open-ended tasks where a single judge might introduce bias.

```yaml
constants:
  judges:
    - name: qwen3
      model: qwen/qwen3-235b-a22b-2507
      api_key: your_openrouter_api_key
      base_url: https://openrouter.ai/api/v1/
    - name: mistral_large
      model: mistralai/mistral-large-2411
      api_key: your_openrouter_api_key
      base_url: https://openrouter.ai/api/v1/
    - name: claude_sonnet
      model: anthropic/claude-3.5-sonnet
      api_key: your_openrouter_api_key
      base_url: https://openrouter.ai/api/v1/

  tasks:
    - name: open_ended_multi_judge
      task_name: open_ended
      num_fewshot: 0
      max_tokens: 10000
      judges: !ref judges  # Use multiple judges
      batch_size: 15
```

**Multi-Judge Metrics:**
- `llm_as_judge_{judge_name}`: Individual score from each judge
- `judge_voting`: Majority vote result (recommended primary metric)
- `llm_as_judge_avg`: Average score across all judges
- `judge_agreement`: Percentage of samples where all judges agree

**Recommendations:**
- Use **3 judges** for a good balance between cost and reliability
- Use **5 judges** for high-stakes evaluations
- Avoid **2 judges** (risk of ties)
- Mix different model architectures and providers for diversity

### Models Configuration

Configure one or more models to evaluate:

```yaml
models:
  - name: model-identifier
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 5      # Concurrent API requests (default: 3)
    timeout: 180          # Request timeout in seconds (default: 300)
    tasks: !ref tasks     # Reference to tasks list
```

#### EVE API Model Configuration

The EVE API provides RAG-enhanced (Retrieval-Augmented Generation) responses with Earth Observation context. To use the EVE API:

```yaml
constants:
  # EVE API credentials
  eve_email: your-email@example.com
  eve_password: your-password
  eve_base_url: http://0.0.0.0:8000/
  eve_public_collections: ['qwen-512-filtered', 'Wikipedia EO', 'Wiley AI Gateway']
  eve_k: 10
  eve_threshold: 0.5

  tasks:
    - name: open_ended_0_shot
      task_name: open_ended
      num_fewshot: 0
      max_tokens: 10000
      model_type: eve-api  # Specify eve-api model type

models:
  - name: eve-api
    # EVE API configuration
    email: !ref eve_email
    password: !ref eve_password
    base_url: !ref eve_base_url
    public_collections: !ref eve_public_collections
    k: !ref eve_k
    threshold: !ref eve_threshold

    # General settings
    temperature: 0.0
    num_concurrent: 7
    tasks: !ref tasks
    timeout: 180
```

**EVE API Parameters:**
- `email`: Email for EVE API authentication
- `password`: Password for EVE API authentication
- `base_url`: Base URL for the EVE API endpoint
- `public_collections`: List of document collections to search for RAG context
- `k`: Number of documents to retrieve (default: 5)
- `threshold`: Similarity threshold for document retrieval (default: 0.5)

**Important Notes:**
- Tasks using EVE API must specify `model_type: eve-api` in the task configuration
- The EVE API automatically retrieves relevant context documents for each query
- Retrieved documents are used to enhance the model's responses
- Especially useful for Earth Observation domain-specific questions
- Requires a running EVE API server at the specified `base_url`

**Example Use Case:**

The EVE API is particularly valuable for:
- Evaluating models on Earth Observation tasks with factual grounding
- Comparing RAG-enhanced responses vs. non-RAG responses
- Testing model performance with domain-specific context retrieval
- Hallucination detection where factual context is critical

### Weights & Biases (WandB) Logging

Enable experiment tracking with WandB:

```yaml
wandb:
  enabled: true                    # Enable/disable WandB logging
  project: project-name            # WandB project name
  entity: organization-name        # WandB entity/organization
  run_name: custom-run-name       # Optional: Custom run name prefix
  api_key: your-wandb-api-key     # WandB API key
```

When enabled, the evaluation will log:
- Evaluation metrics (accuracy, F1, IoU, etc.)
- Individual sample predictions
- Task configurations
- Model metadata
- Evaluation duration and timestamps

### Output Directory

Specify where evaluation results should be saved:

```yaml
output_dir: evals_outputs  # Default: eval_results
```

---

## Example Configurations

For comprehensive configuration examples including:
- EVE Earth Observation tasks
- LM-Eval-Harness standard benchmarks
- Mixed evaluations
- Multiple model comparisons
- Environment variables usage
- Testing with limited samples

Please see the dedicated **[Examples](examples.md)** page.

---

## Output Structure

After running evaluations, results are saved organized by task, then by model:

```
{output_dir}/
├── {task_name_1}/
│   ├── {model_name_sanitized}/
│   │   ├── results_{timestamp}.json
│   │   └── samples_{task_name}_{timestamp}.jsonl
│   ├── {another_model_name_sanitized}/
│   │   ├── results_{timestamp}.json
│   │   └── samples_{task_name}_{timestamp}.jsonl
│   └── ...
├── {task_name_2}/
│   └── ...
```

### Example Structure:

```
evals_outputs/
├── hallucination_detection/
│   ├── eve-esa__eve_v0.1/
│   │   ├── results_2025-12-01T10-17-45.479920.json
│   │   └── samples_hallucination_detection_2025-12-01T10-17-45.479920.jsonl
│   └── gpt-4/
│       ├── results_2025-12-01T10-20-15.123456.json
│       └── samples_hallucination_detection_2025-12-01T10-20-15.123456.jsonl
├── mcqa_single_answer/
│   ├── eve-esa__eve_v0.1/
│   │   ├── results_2025-12-01T11-23-12.123456.json
│   │   └── samples_mcqa_single_answer_2025-12-01T11-23-12.123456.jsonl
│   └── gpt-4/
│       ├── results_2025-12-01T11-25-30.789012.json
│       └── samples_mcqa_single_answer_2025-12-01T11-25-30.789012.jsonl
└── open_ended/
    ├── eve-esa__eve_v0.1/
    │   ├── results_2025-12-01T12-34-56.789012.json
    │   └── samples_open_ended_2025-12-01T12-34-56.789012.jsonl
    └── gpt-4/
        ├── results_2025-12-01T12-40-10.456789.json
        └── samples_open_ended_2025-12-01T12-40-10.456789.jsonl
```

This structure makes it easy to compare multiple models on the same task.

### Results File Format

The `results_{timestamp}.json` file contains:

```json
{
  "results": {
    "task_name": {
      "alias": "task_name",
      "metric_1,none": 0.85,
      "metric_1_stderr,none": 0.02,
      "metric_2,none": 0.78,
      "metric_2_stderr,none": 0.03
    }
  },
  "group_subtasks": {},
  "configs": {
    "task_name": {
      "task": "task_name",
      "dataset_path": "dataset-path",
      "num_fewshot": 0,
      "metadata": {}
    }
  },
  "versions": {},
  "n-shot": {},
  "n-samples": {},
  "config": {},
  "git_hash": "abc123",
  "date": 1701234567.89,
  "total_evaluation_time_seconds": "123.45"
}
```

### Samples File Format

The `samples_{task_name}_{timestamp}.jsonl` file contains individual predictions:

```jsonl
{"doc_id": 0, "doc": {...}, "target": "expected", "arguments": [...], "resps": [["predicted"]], "filtered_resps": ["predicted"], "doc_hash": "abc123", "prompt_hash": "def456", "task_name": "task_name"}
{"doc_id": 1, "doc": {...}, "target": "expected", "arguments": [...], "resps": [["predicted"]], "filtered_resps": ["predicted"], "doc_hash": "ghi789", "prompt_hash": "jkl012", "task_name": "task_name"}
...
```

Each line contains:
- `doc`: The input document/question
- `target`: Expected answer
- `resps`: Raw model response
- `filtered_resps`: Processed model response
- Metadata for reproducibility

---

## WandB Integration

When WandB logging is enabled, the following information is automatically logged:

### Metrics Logged

- **Aggregate Metrics**: Final scores for each metric (accuracy, F1, IoU, etc.)
- **Per-Sample Metrics**: Individual predictions and correctness
- **Task Metadata**: Dataset paths, splits, versions
- **Model Configuration**: API endpoints, temperatures, timeouts
- **Evaluation Metadata**: Git hash, timestamps, duration

### Viewing Results

After evaluation completes, visit your WandB project to:

1. **Compare Models**: View metrics across different models side-by-side
2. **Analyze Samples**: Inspect individual predictions and failures
3. **Track Progress**: Monitor evaluation progress in real-time
4. **Visualize Trends**: Plot metric distributions and comparisons

### Example WandB Output

```
Run: eve-model-v1-20251201
├── Summary Metrics
│   ├── hallucination_detection/acc: 0.822
│   ├── hallucination_detection/f1: 0.841
│   ├── hallucination_detection/precision: 0.869
│   ├── mcqa_single_answer/acc: 0.756
│   └── open_ended/llm_judge: 0.834
├── Config
│   ├── model: eve-esa/eve_v0.1
│   ├── temperature: 0.1
│   └── num_concurrent: 10
└── Samples
    ├── hallucination_detection_samples.csv
    ├── mcqa_single_answer_samples.csv
    └── open_ended.csv
```

---

## Advanced Usage

### Using Environment Variables

Instead of hardcoding API keys, use environment variables. See [Examples](examples.md#example-5-using-environment-variables) for detailed configuration.

### Limiting Samples for Testing

Test your configuration on a small subset. See [Examples](examples.md#example-6-testing-with-limited-samples) for detailed configuration.

### Direct Command Line 

For quick tests, you can use the lm_eval command directly:

```bash
lm_eval --model openai-chat-completions \
        --model_args base_url=https://api.provider.com,model=model-name,num_concurrent=5 \
        --tasks hallucination_detection,mcqa_single_answer \
        --include tasks \
        --num_fewshot 0 \
        --output_path ./outputs \
        --log_samples \
        --apply_chat_template
```

---

## Available Tasks

### EVE Earth Observation Tasks

See the [EO Tasks](eo_tasks.md) page for detailed information about:
- `mcqa_multiple_answer`
- `mcqa_single_answer`
- `open_ended`
- `open_ended_w_context`
- `refusal`
- `hallucination_detection`


### LM-Evaluation-Harness Tasks

All tasks from the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) are supported, including:

**Popular Benchmarks:**
- `mmlu_pro` - MMLU-Pro (challenging multiple-choice)
- `gsm8k` - Grade School Math
- `hellaswag` - Commonsense reasoning
- `arc_challenge` - AI2 Reasoning Challenge
- `truthfulqa` - Truthfulness evaluation
- `winogrande` - Commonsense reasoning
- `piqa` - Physical commonsense
- And more...

**To list all available tasks:**

```bash
lm_eval --tasks list
```

---

## Troubleshooting

### Common Issues

**1. API Timeout Errors**

Increase the timeout value:

```yaml
models:
  - name: your-model
    timeout: 600  # Increase to 10 minutes
```

**2. Rate Limiting**

Reduce concurrent requests:

```yaml
models:
  - name: your-model
    num_concurrent: 1  # Reduce concurrency
```

**3. Judge Model Errors**

Ensure judge credentials are set for tasks that require them:

```yaml
tasks:
  - name: open_ended
    judge_api_key: !ref judge_api_key  # Required!
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**4. WandB Login Issues**

Login before running:

```bash
wandb login your-api-key
```

---

## Next Steps

- **Explore Tasks**: Check out the [EO Tasks](eo_tasks.md) page for details on Earth Observation evaluation tasks

## Support

For issues or questions:
- GitHub Issues: [eve-esa/evalkit](https://github.com/eve-esa/evalkit/issues)
- Documentation: [https://docs.eve-evaluation.org](https://docs.eve-evaluation.org)
- LM-Eval-Harness: [EleutherAI Documentation](https://github.com/EleutherAI/lm-evaluation-harness)
