# Earth Observation Evaluation Tasks

This page provides a comprehensive overview of all available Earth Observation (EO) evaluation tasks in eve-evalkit. Each task is designed to assess different capabilities of language models in the Earth Observation domain.

## Quick Reference

| Task Name | Type | Dataset | Size | Primary Metrics |
|-----------|------|---------|------|-----------------|
| [MCQA Multiple Answer](#mcqa-multiple-answer) | Multiple Choice | [eve-esa/eve-is-mcqa](https://huggingface.co/datasets/eve-esa/eve-is-mcqa) | 431  | IoU, Accuracy |
| [MCQA Single Answer](#mcqa-single-answer) | Multiple Choice | [eve-esa/mcqa-single-answer](https://huggingface.co/datasets/eve-esa/mcqa-single-answer) | 1261 | Accuracy |
| [Open Ended](#open-ended) | Generation | [eve-esa/open-ended](https://huggingface.co/datasets/eve-esa/open-ended) | 1257 | LLM as Judge, Win Rate |
| [Open Ended with Context](#open-ended-with-context) | Generation | [eve-esa/open-ended-w-context](https://huggingface.co/datasets/eve-esa/open-ended-w-context) | 418  | LLM Judge, Win Rate |
| [Hallucination Detection](#hallucination-detection) | Classification | [eve-esa/hallucination-detection](https://huggingface.co/datasets/eve-esa/hallucination-detection) | 2326 | Accuracy, Precision, Recall, F1 |

---

## Detailed Task Descriptions


### MCQA Multiple Answer

**Task Name:** `is_mcqa` or `mcqa_multiple_answer`

**Description:**

EVE-mcqa-multiple-answers consists of multiple-choice questions from Imperative Space MOOC exams where questions may have one or more correct answers. Models must identify all correct options from an arbitrary number of choices, making this a challenging task that requires comprehensive understanding rather than simple fact recall.

**How to Call:**

```yaml
tasks:
  - name: mcqa_multiple_answers
    num_fewshot: 2
    max_tokens: 10000
```

**Dataset:**

- **Source:** [eve-esa/mcqa-multiple-answers](https://huggingface.co/datasets/eve-esa/mcqa-multiple-answers)
- **Split:** train
- **Size:** 432 samples
- **Structure:** Each example contains a `Question`, `Answers` (list of correct labels), and `Choices` (list with labels and text)

**Evaluation Metrics:**

- **IoU (Intersection over Union)**: Measures partial correctness by calculating the overlap between predicted and correct answer sets. IoU = |Predicted ∩ Correct| / |Predicted ∪ Correct| (higher is better)
- **Accuracy (Exact Match)**: Binary score where 1.0 means the predicted answer set exactly matches the correct answer set (higher is better)

**Why It's Useful:**

This task tests a model's comprehensive understanding of EO concepts where multiple aspects or factors may be simultaneously correct. The IoU metric is particularly valuable as it rewards partially correct answers, providing a more nuanced evaluation than simple exact matching. This reflects real-world scenarios where partial knowledge is still valuable.

**Example:**

```python
{
  "Question": "Which bands of Sentinel-2 have 10m resolution?",
  "Answers": ["A", "B", "C"],
  "Choices": {
    "label": ["A", "B", "C", "D"],
    "text": ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"]
  }
}
```

---

### MCQA Single Answer

**Task Name:** `mcqa_single_answer`

**Description:**

EVE-mcqa-single-answer is a traditional multiple-choice dataset with exactly one correct answer per question. Models must identify the single best option from the provided choices, testing factual knowledge and reasoning abilities in the Earth Observation domain.

**How to Call:**

```yaml
tasks:
  - name: mcqa_single_answer
    num_fewshot: 2
    max_tokens: 10000
```

**Dataset:**

- **Source:** [eve-esa/mcqa-single-answer](https://huggingface.co/datasets/eve-esa/mcqa-single-answer)
- **Split:** train
- **Size:** ~1000 samples
- **Structure:** Each example contains a `question`, `choices` (list of answer texts), and `answer` (single letter indicating correct choice)

**Evaluation Metrics:**

- **Accuracy**: Percentage of questions answered correctly (higher is better)

**Why It's Useful:**

This task evaluates factual knowledge and reasoning abilities in scenarios where there is a single definitively correct answer. It's particularly useful for assessing fundamental EO concepts, terminology, and principles. The single-answer format reduces ambiguity and provides clear, interpretable results.

**Example:**

```python
{
  "question": "What is the spatial resolution of Sentinel-2's visible bands?",
  "choices": ["5 meters", "10 meters", "20 meters", "60 meters"],
  "answer": "B"  # "10 meters"
}
```

---

### Open Ended

**Task Name:** `open_ended`

**Description:**

EVE-open-ended is a collection of ~969 open-ended question-answer pairs focused on Earth Observation. The dataset covers a wide range of EO topics including satellite imagery analysis, remote sensing techniques, environmental monitoring, and LiDAR. Models must generate free-form responses demonstrating deep understanding without the constraints of multiple-choice formats.

**How to Call:**

```yaml
tasks:
  - name: open_ended
    num_fewshot: 5
    max_tokens: 40000
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/open-ended](https://huggingface.co/datasets/eve-esa/open-ended)
- **Split:** train
- **Size:** ~969 samples
- **Structure:** Each example contains a `Question` and `Answer` (reference)

**Evaluation Metrics:**

- **LLM as Judge**: A judge model evaluates the quality, accuracy, and completeness of generated answers using strict fact-checking rules (0 = FAIL, 1 = PASS)
- **Multi-Judge Support**: Use multiple judges for more robust evaluation (see Multi-Judge Evaluation section below)
- **Win Rate**: Compare two models head-to-head using multiple LLM judges (see Win Rate Evaluation section below)
- Alternative metrics: BLEU, ROUGE, Cosine Similarity, BERTScore

**LLM Judge Evaluation Rules:**

1. **Contradiction Check**: Fails if the answer contains ANY fact contradicting the reference
2. **Relevance Check**: Fails if the answer omits ESSENTIAL technical facts from the reference
3. **Additive Information**: Additional correct information is acceptable if it doesn't contradict
4. **Focus on Substance**: Ignores style, length, and tolerates minor phrasing differences

**Why It's Useful:**

This task assesses a model's ability to explain concepts, provide detailed answers, and demonstrate deep understanding. It's essential for evaluating models intended for educational or explanatory applications in EO, where nuanced explanations and technical accuracy are paramount.

---

### Open Ended with Context

**Task Name:** `open_ended_w_context`

**Description:**

EVE-open-ended-w-context provides open-ended questions that must be answered using 1-3 accompanying context documents. This tests the model's ability to extract and synthesize information from reference materials, making it ideal for evaluating Retrieval-Augmented Generation (RAG) systems. Not all samples contain all three documents, requiring models to handle variable numbers of context documents gracefully.

**How to Call:**

```yaml
tasks:
  - name: open_ended_w_context
    num_fewshot: 5
    max_tokens: 40000
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/open-ended-w-context](https://huggingface.co/datasets/eve-esa/open-ended-w-context)
- **Split:** train
- **Structure:** Each example contains a `Question`, `Answer`, and up to three context documents (`Doc 1`, `Doc 2`, `Doc 3`)

**Evaluation Metrics:**

- **LLM Judge**: Evaluates whether answers are grounded in the provided context and correctly answer the question (higher is better)
- **Multi-Judge Support**: Use multiple judges for more robust evaluation (see Multi-Judge Evaluation section below)
- **Win Rate**: Compare two models head-to-head using multiple LLM judges (see Win Rate Evaluation section below)
- Uses the same strict fact-checking evaluation rules as open-ended tasks

**Why It's Useful:**

This task evaluates retrieval-augmented generation (RAG) capabilities, testing whether models can accurately extract information from provided documents rather than relying solely on parametric knowledge. This is crucial for applications where answers must be grounded in specific documentation or data sources. It also tests the model's ability to distinguish between context-provided information and pre-trained knowledge.

**Example:**

```python
{
  "Question": "What is the spatial resolution of Sentinel-2's visible bands?",
  "Answer": "Sentinel-2's visible bands have a spatial resolution of 10 meters.",
  "Doc 1": "The Sentinel-2 mission comprises a constellation...",
  "Doc 2": "Sentinel-2 carries the Multi-Spectral Instrument (MSI)...",
  "Doc 3": ""  # May be empty
}
```

---

### Refusal

**Task Name:** `refusal`

**Description:**

EVE-Refusal tests whether language models can appropriately refuse to answer questions when the provided context does not contain sufficient information. The dataset presents questions alongside context documents that intentionally lack the necessary information to answer. A well-calibrated model should recognize this limitation and refuse to answer, rather than generating plausible but incorrect information.

**How to Call:**

```yaml
tasks:
  - name: refusal
    num_fewshot: 5
    max_tokens: 40000
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/refusal](https://huggingface.co/datasets/eve-esa/refusal)
- **Split:** train
- **Structure:** Each example contains a `question` and `context` (insufficient for answering)
- **Expected Answer:** "I'm sorry, but the provided context does not contain enough information to answer that question."

**Evaluation Metrics:**

- **LLM Judge**: Evaluates whether the model appropriately refuses to answer or acknowledges insufficient information (higher is better)

**Expected Behavior:**

- Recognize when provided context lacks sufficient information
- Explicitly refuse to answer or state information is not available
- Avoid generating plausible-sounding but fabricated information
- Maintain accuracy and honesty over completeness

**Why It's Useful:**

This task tests a critical safety and reliability feature: the ability to recognize limitations and avoid generating potentially incorrect information when context is insufficient. This prevents hallucinations and ensures trustworthy behavior in production systems. It's particularly important for RAG systems and applications where factual accuracy is paramount.

---

### Hallucination Detection

**Task Name:** `hallucination_detection`

**Description:**

EVE-Hallucination is a specialized dataset for evaluating language models' tendency to hallucinate in the Earth Observation domain. Unlike typical QA datasets, this contains deliberately hallucinated answers with detailed annotations marking which portions of text are hallucinated. The task is to identify whether a given answer contains hallucinated (false or unsupported) information.

**How to Call:**

```yaml
tasks:
  - name: hallucination_detection
    num_fewshot: 0
    max_tokens: 100
    judge_api_key: !ref judge_api_key
    judge_base_url: !ref judge_base_url
    judge_name: !ref judge_name
```

**Dataset:**

- **Source:** [eve-esa/hallucination-detection](https://huggingface.co/datasets/eve-esa/hallucination-detection)
- **Split:** train
- **Structure:** Each example contains `Question`, `Answer` (with hallucinations), `Soft labels` (probabilistic spans), and `Hard labels` (definite spans)

**Evaluation Metrics:**

- **Accuracy**: Overall correctness of hallucination detection (higher is better)
- **Precision**: Ratio of correctly identified hallucinations to all predicted hallucinations (higher is better)
- **Recall**: Ratio of correctly identified hallucinations to all actual hallucinations (higher is better)
- **F1 Score**: Harmonic mean of precision and recall (higher is better)

**Task Levels:**

1. **Binary Detection**: Determine if answer contains any hallucinated information (yes/no)
2. **Hard Span Detection**: Identify exact character spans that are hallucinated
3. **Soft Span Detection**: Identify spans with confidence scores

**Why It's Useful:**

This task evaluates a model's ability to self-assess and identify unreliable or fabricated information in EO contexts. Models with strong hallucination detection capabilities are more trustworthy and can potentially be used to validate outputs from other systems. This is crucial for safety-critical applications like climate monitoring, disaster response, and environmental analysis.

**Example:**

```python
{
  "Question": "What is the spatial resolution of Sentinel-2's visible bands?",
  "Answer": "Sentinel-2's visible bands have a spatial resolution of 5 meters, making it the highest resolution freely available satellite.",
  "Hard labels": [[52, 60], [73, 127]]  # Character spans that are hallucinated
}
```

---

## Multi-Judge Evaluation

For open-ended tasks (`open_ended`, `open_ended_w_context`, `open_ended_w_context_full`), you can use **multi-judge evaluation** where multiple LLM judges independently evaluate each answer. This approach provides more robust and reliable evaluation through consensus-based scoring.

### Benefits

- **Reduced Bias**: Individual judge biases are averaged out across multiple judges
- **Voting Metric**: Majority vote provides a robust, democratic final score
- **Agreement Tracking**: Monitor consensus to identify ambiguous or controversial samples
- **Judge Analysis**: Compare individual judges to identify systematic differences or biases

### Configuration

Define multiple judges in your `evals.yaml` configuration:

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
      model_type: local-chat-completions
      num_fewshot: 0
      max_tokens: 10000
      judges: !ref judges  # Use all judges defined above
      batch_size: 15
```

### Metrics Produced

When using multi-judge evaluation, the following metrics are automatically generated:

#### 1. Individual Judge Scores
- `llm_as_judge_{judge_name}`: Score from each individual judge (e.g., `llm_as_judge_qwen3`, `llm_as_judge_mistral_large`)
- Values: 0 or 1
- Use to identify systematic differences between judges

#### 2. Voting Metric (Recommended)
- `judge_voting`: **Majority vote result**
- Returns the score that has majority support (>= half+1 judges)
- For ties (even number of judges with equal votes), defaults to 0
- Values: 0 or 1
- **This is the recommended primary metric**

#### 3. Average Score
- `llm_as_judge_avg`: Average score across all judges
- Values: 0.0 to 1.0
- Provides granular scores useful for ranking models

#### 4. Agreement Metric
- `judge_agreement`: Percentage of samples where all judges agree
- Values: 0.0 to 1.0
- High agreement (>0.8) indicates judges are consistent
- Low agreement (<0.5) suggests ambiguous questions or edge cases

### Voting Examples

**With 2 judges:**
- Both vote 1 → voting = 1
- Both vote 0 → voting = 0
- 1 votes 1, 1 votes 0 → voting = 0 (tie, no majority)

**With 3 judges:**
- 2 vote 1, 1 votes 0 → voting = 1 (majority)
- 1 votes 1, 2 vote 0 → voting = 0 (majority)
- All vote 1 → voting = 1 (unanimous)

**With 5 judges:**
- 3 vote 1, 2 vote 0 → voting = 1 (majority)
- 2 vote 1, 3 vote 0 → voting = 0 (majority)

### Recommendations

**Number of Judges:**
- **3 judges**: Good balance between cost and reliability (recommended)
- **5 judges**: Better for high-stakes evaluations
- **2 judges**: Avoid if possible (risk of ties with no clear majority)

**Judge Selection:**
- Use diverse models (different architectures/providers)
- Mix model sizes (small + large models)
- Include both specialized and general-purpose models
- Example: Claude, GPT-4, Mistral Large, Qwen

**Cost Optimization:**
1. Start with 3 judges on a small sample (limit: 10-50)
2. Analyze the agreement rate
3. If agreement is high (>0.8), consider using single judge or voting metric
4. If agreement is low (<0.5), investigate question quality or add more judges

### Example Output

```json
{
  "results": {
    "open_ended": {
      "llm_as_judge_qwen3": 0.75,
      "llm_as_judge_mistral_large": 0.80,
      "llm_as_judge_claude_sonnet": 0.78,
      "llm_as_judge_avg": 0.777,
      "judge_agreement": 0.65,
      "judge_voting": 0.80
    }
  }
}
```

In this example:
- Individual judges scored 75%, 80%, and 78%
- Overall average is 77.7%
- Judges fully agreed on 65% of samples
- **Majority vote gave 80% (recommended metric to report)**

---

## Win Rate Evaluation

For open-ended tasks (`open_ended`, `open_ended_w_context`), you can perform **win rate evaluation** to compare two models head-to-head using multiple LLM judges. This separate evaluation script provides comparative analysis between model outputs, complementing the standard LLM-as-judge metrics.

### What is Win Rate?

Win rate evaluation compares the outputs of two models (Model A vs Model B) on the same questions and determines which model provides better answers according to independent LLM judges. This approach is particularly useful for:

- **Model Selection**: Directly compare two models to identify which performs better
- **Model Improvement**: Assess whether a new model version improves over a baseline
- **Ablation Studies**: Evaluate the impact of specific model changes or training approaches
- **Benchmark Comparison**: Compare your model against established baselines or competitors

### Key Differences from Standard Evaluation

The win rate evaluation is a **separate script** with its own configuration format and workflow:

| Aspect | Standard Evaluation | Win Rate Evaluation |
|--------|---------------------|---------------------|
| **Purpose** | Evaluate single model quality | Compare two models head-to-head |
| **Script** | `scripts/evaluate.py` | `metrics/win_rate/win_rate_evaluation.py` |
| **Configuration** | `evals.yaml` | Separate YAML config (e.g., `win_rate_config.yaml`) |
| **Input** | Live model API | Pre-generated CSV files from standard evaluation |
| **Output Format** | Standard metrics (accuracy, F1) | Win rates, alpaca win rates, judge agreement |
| **Judges** | Single or multi-judge per eval | Multiple judges comparing two outputs |
| **WandB Logging** | Evaluation results | Win rate metrics, visualizations, judge rationales |

### Metrics Explained

#### 1. Win Rate

Percentage of questions where each model won according to each judge.

**Formula:**
```
Win Rate = (Number of Wins / Total Evaluations) × 100%
```

**Calculation Process:**
- For each question, each judge compares Model A vs Model B outputs
- Judge decides: Model A wins, Model B wins, or Tie
- Win rate = (wins / total evaluations) × 100%
- Aggregate win rate = average win rate across all judges

**Interpreting Win Rate Values:**
- **0.50 (50%)**: Models perform equally well (perfect tie across all questions)
- **> 0.50**: Model is better than its competitor
  - 0.55-0.60: Slight advantage
  - 0.60-0.70: Clear advantage
  - 0.70+: Strong advantage
- **< 0.50**: Model is worse than its competitor
- **Win Rate Difference**: The gap between Model A and Model B
  - Difference < 0.05: Negligible difference
  - Difference 0.05-0.10: Noticeable difference
  - Difference > 0.10: Significant performance gap

**Logged Metrics:**
- `win_rate/{judge_name}/{model_name}`: Win rate for each model per judge (0.0 to 1.0)
- `aggregate/{model_name}_win_rate`: Aggregate win rate across all judges
- `aggregate/avg_{model_name}_win_rate`: Average win rate across judges
- `aggregate/win_rate_difference`: Difference between Model A and Model B win rates

**Example:**
```
Model A wins: 72 questions
Model B wins: 25 questions
Ties: 3 questions
Total: 100 questions

Model A Win Rate = 72/100 = 0.72 (72%)
Model B Win Rate = 25/100 = 0.25 (25%)
Win Rate Difference = 0.72 - 0.25 = 0.47 (47% gap - significant advantage for Model A)
```

#### 2. Alpaca Win Rate

A more nuanced metric that counts ties as half a win for each model, based on the [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) framework.

**Formula:**
```
Alpaca Win Rate = (Number of Wins + 0.5 × Number of Ties) / Total Evaluations
```

**Why Use Alpaca Win Rate?**
- Treats ties as split decisions, giving partial credit to both models
- More granular than standard win rate when there are many ties
- Better reflects cases where models perform similarly on some questions
- Recommended by AlpacaEval for instruction-following model comparisons

**Interpreting Alpaca Win Rate Values:**
- **0.50 (50%)**: Models perform equally well
- **> 0.50**: Model is better (same interpretation as standard win rate)
- Alpaca win rate is always ≥ standard win rate (due to partial credit for ties)
- Use alpaca win rate when you have many ties and want more nuanced comparison

**Logged Metrics:**
- `alpaca_win_rate/{judge_name}/{model_name}`: Alpaca win rate per judge
- `aggregate/{model_name}_alpaca_win_rate`: Aggregate alpaca win rate
- `aggregate/avg_{model_name}_alpaca_win_rate`: Average alpaca win rate across judges
- `aggregate/alpaca_win_rate_difference`: Difference between Model A and Model B alpaca win rates

**Example:**
```
Model A wins: 72 questions
Model B wins: 25 questions
Ties: 3 questions
Total: 100 questions

Model A Alpaca Win Rate = (72 + 0.5×3)/100 = 73.5/100 = 0.735 (73.5%)
Model B Alpaca Win Rate = (25 + 0.5×3)/100 = 26.5/100 = 0.265 (26.5%)

Note: Both models get partial credit for the 3 ties
```

**Reference:** [AlpacaEval - Automatic Evaluator for Instruction-following LLMs](https://github.com/tatsu-lab/alpaca_eval)

#### 3. Judge Agreement

Measures how consistently judges agree on which model is better:

- **Unanimous (1.0)**: All judges made the same decision
- **Majority (0.5-0.99)**: Most judges agreed on winner
- **Split (< 0.5)**: Judges were evenly divided

**Interpreting Agreement:**
- **> 0.80**: High agreement - clear quality difference or consistent evaluation
- **0.50-0.80**: Moderate agreement - some subjective variation among judges
- **< 0.50**: Low agreement - questions may be ambiguous or judges have different criteria

#### 4. Position Bias

Analysis of whether judges are biased toward answers shown in position A vs B. The evaluation randomizes answer positions to mitigate this bias.

**What to Look For:**
- Position bias close to 0.50 (50%) indicates no position bias
- Significant deviation from 0.50 suggests judges prefer one position
- Randomization helps ensure fair comparison despite any position bias

### How to Run Win Rate Evaluation

#### Step 1: Generate Model Outputs

First, run standard evaluation to generate CSV files with model outputs:

```bash
python scripts/evaluate.py evals.yaml
```

This creates CSV files in your output directory (e.g., `evals_outputs/{model_name}/samples_open_ended.csv`) with columns:
- `doc\.Question`: The question text
- `target`: The reference/ground truth answer
- `filtered_resps`: The model's response

#### Step 2: Create Win Rate Configuration

Create a YAML configuration file (e.g., `win_rate_config.yaml`):

```yaml
# Task type for metrics logging
task: "open_ended"  # or "open_ended_w_context"

# Models to compare
model_a:
  - name: "model-a-name"
    file: "path/to/model_a_output.csv"

model_b:
  - name: "model-b-name"
    file: "path/to/model_b_output.csv"

# LLM Judges configuration
judges:
  - name: "mistral-large"
    model: "mistral-large-2512"
    api_key: "${MISTRAL_API_KEY}"
    base_url: "https://api.mistral.ai/v1"

  - name: "gpt-4-mini"
    model: "openai/gpt-4.1-mini"
    api_key: "${OPENROUTER_API_KEY}"
    base_url: "https://openrouter.ai/api/v1/"

  - name: "qwen3-235b"
    model: "qwen/qwen3-235b-a22b-2507"
    api_key: "${OPENROUTER_API_KEY}"
    base_url: "https://openrouter.ai/api/v1/"

# Evaluation settings
evaluation:
  limit: null  # Set to N to limit to first N questions, or null for all
  max_workers: 20  # Number of parallel threads
  rate_limit_delay: 0.05  # Delay between API calls (seconds)
  random_seed: 42  # For reproducibility (null = random)

# Output settings
output:
  save_results: true
  save_visualizations: true
  output_dir: "win_rate_results"
  results_filename: "results_{model_a}_vs_{model_b}.csv"
  summary_filename: "summary_{model_a}_vs_{model_b}.csv"
  visualization_filename: "comparison_{model_a}_vs_{model_b}.png"

# Weights & Biases configuration
wandb:
  enabled: true
  project: "eve-win-rate-evaluation"
  entity: your-wandb-username  # or null for default
  run_name: "{model_a}_vs_{model_b}"
  tags:
    - "win-rate"
    - "llm-judge"

  # What to log
  log:
    win_rates: true
    accuracy_rates: true
    judge_agreement: true
    position_bias: true
    visualizations: true
    raw_results: true
    sample_rationales: true
    sample_count: 5
```

#### Step 3: Run Win Rate Evaluation

```bash
python metrics/win_rate/win_rate_evaluation.py --config win_rate_config.yaml
```

#### Step 4: View Results

The script generates:

1. **CSV Results** (`win_rate_results/results_*.csv`): Complete evaluation data with all judge decisions
2. **Summary CSV** (`win_rate_results/summary_*.csv`): Win rate statistics per judge
3. **Visualizations** (`win_rate_results/comparison_*.png`): Charts showing win rates
4. **WandB Dashboard**: Interactive results with metrics, visualizations, and sample rationales

### Comparing Multiple Models

You can compare multiple models pairwise by configuring lists for `model_a` and `model_b`:

```yaml
model_a:
  - name: "eve_v04"
    file: "generations/open_ended/scoring_eve_v04_open_ended_0_shot.csv"
  - name: "eve_v05"
    file: "generations/open_ended/scoring_eve_v05_open_ended_0_shot.csv"

model_b:
  - name: "mistral-small"
    file: "generations/open_ended/scoring_mistral-small_open_ended_0_shot.csv"
  - name: "llama-4-scout"
    file: "generations/open_ended/scoring_llama-4-scout_open_ended_0_shot.csv"
```

This will run all pairwise comparisons: eve_v04 vs mistral-small, eve_v04 vs llama-4-scout, eve_v05 vs mistral-small, eve_v05 vs llama-4-scout.

### Best Practices

1. **Number of Judges**: Use 3-5 judges for robust evaluation
2. **Judge Diversity**: Select judges from different model families (e.g., GPT, Claude, Mistral, Qwen)
3. **Rate Limiting**: Adjust `rate_limit_delay` if hitting API rate limits
4. **Reproducibility**: Set `random_seed` to a fixed value for reproducible position randomization
5. **Testing**: Start with `limit: 10` to test configuration before running full evaluation
6. **WandB Tracking**: Enable wandb logging to track experiments and compare runs

### Example Output

After running win rate evaluation, you'll see output like:

```
Model A: eve_v05
Model B: mistral-small-3.2-24b

=== Aggregate Results ===
eve_v05:
  Win Rate: 0.72 (72%)
  Alpaca Win Rate: 0.75 (75%)

mistral-small-3.2-24b:
  Win Rate: 0.28 (28%)
  Alpaca Win Rate: 0.25 (25%)

Judge Agreement: 0.68 (68% unanimous decisions)

WandB Run: https://wandb.ai/your-entity/eve-win-rate-evaluation/runs/...
```

**Interpretation**: eve_v05 clearly outperforms mistral-small-3.2-24b with a 44% win rate gap (0.72 - 0.28), indicating strong superiority across the evaluated questions.

### References

- **AlpacaEval Framework**: [https://github.com/tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval)
- **Full Documentation**: See `WIN_RATE_EVALUATION_README.md` in the repository root
- **Example Configurations**: See `metrics/win_rate/win_rate_open_ended_example.yaml`

---

## Running Tasks

### Using Configuration File

Add tasks to your `evals.yaml`:

```yaml
constants:
  judge_api_key: your-judge-api-key
  judge_base_url: https://openrouter.ai/api/v1
  judge_name: mistralai/mistral-large-2411
  tasks:
    - name: mcqa_multiple_answers
      num_fewshot: 2
      max_tokens: 10000
    - name: hallucination_detection
      num_fewshot: 0
      max_tokens: 100

models:
  - name: your-model-name
    base_url: https://api.provider.com/v1/chat/completions
    api_key: your-api-key
    temperature: 0.1
    num_concurrent: 5
    tasks: !ref tasks

output_dir: evals_outputs
```

Then run:

```bash
python scripts/evaluate.py evals.yaml
```

### Direct Command Line

```bash
lm_eval --model openai-chat-completions \
        --model_args base_url=https://api.provider.com,model=model-name,num_concurrent=5 \
        --tasks {task_name} \
        --include tasks \
        --num_fewshot 0 \
        --output_path ./outputs \
        --log_samples \
        --apply_chat_template
```

For tasks using LLM-as-judge metrics, set environment variables:

```bash
export JUDGE_API_KEY=your-judge-api-key
export JUDGE_BASE_URL=https://api.provider.com/v1
export JUDGE_NAME=judge-model-name
```

---

## Task Selection Guide

Choose tasks based on your evaluation goals:

**Factual Knowledge:**
- `mcqa_single_answer` - Single correct answer questions
- `mcqa_multiple_answers` - Multiple correct answers with partial credit

**Generation Quality:**
- `open_ended` - Free-form explanatory answers

**Grounded Generation (RAG):**
- `open_ended_w_context` - Answer questions using provided documents
- `refusal` - Recognize when context is insufficient

**Reliability & Safety:**
- `hallucination_detection` - Identify fabricated information
- `refusal` - Avoid answering without sufficient information

**Comprehensive Evaluation:**
- Run all tasks for a complete assessment across different capabilities

---

## Evaluation Best Practices

1. **Use Few-Shot Examples**: Most tasks benefit from few-shot examples (typically 2-5) to demonstrate the expected format
2. **Set Appropriate Timeouts**: Some tasks require longer generation, so adjust timeouts accordingly
3. **Configure Judge Model**: For LLM-as-judge tasks, choose a capable judge model (e.g., GPT-4, Claude 3.5 Sonnet, Mistral Large)
4. **Log Samples**: Always use `--log_samples` to inspect individual predictions and understand model behavior
5. **Monitor Costs**: LLM-as-judge evaluation can be expensive; consider using smaller subsets for initial testing

---

## Additional Resources

- **Dataset Repository:** [https://huggingface.co/eve-esa](https://huggingface.co/eve-esa)
- **GitHub Repository:** [https://github.com/eve-esa/eve-evaluation](https://github.com/eve-esa/eve-evaluation)
- **LM Evaluation Harness:** [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

---

## Citation

If you use these tasks or datasets in your research, please cite:

```bibtex
@misc{eve2025,
  title={EVE: Earth Virtual Expert},
  author={ESA},
  year={2025},
  publisher={HuggingFace},
  url={https://huggingface.co/eve-esa/eve_v0.1}
}
```

For the underlying evaluation framework:

```bibtex
@software{eval-harness,
  author       = {Gao, Leo and others},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```