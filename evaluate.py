import argparse
import dataclasses
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
import wandb


@dataclasses.dataclass
class FieldReference:
    reference_name: str


@dataclasses.dataclass
class FieldWithMaybeType:
    new_type: type
    name: str


@dataclasses.dataclass
class TaskConfig:
    name: str  # User-defined name for this task configuration (used for output folders and wandb)
    task_name: str | None = None  # Actual lm_eval task name (defaults to name if not provided)
    max_tokens: int = 512
    num_fewshot: int = 0
    temperature: float = 0.0
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_name: str = ""
    limit: int | None = None  # Limit number of samples to evaluate

    def __post_init__(self):
        # If task_name is not provided, use name as task_name
        if self.task_name is None:
            self.task_name = self.name


@dataclasses.dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    tasks: list[TaskConfig]
    temperature: float = 0.0
    num_concurrent: int = 3  # Number of concurrent API requests
    timeout: int = 300  # Timeout in seconds (default: 300s / 5 minutes)


@dataclasses.dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "eve-evalkit"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)
    api_key: str | None = None


@dataclasses.dataclass
class EvaluationConfig:
    models: list[ModelConfig]
    output_dir: str = "eval_results"
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    hf_token: str | None = None


def _register_yaml_tags():
    def yaml_reference_tag_constructor(_: yaml.SafeLoader, node: yaml.Node) -> FieldReference:
        if not isinstance(node, yaml.ScalarNode):
            raise NotImplementedError
        if node.tag != "!ref":
            raise NotImplementedError(f"Tag {node.tag} is not supported")
        return FieldReference(reference_name=node.value)

    def yaml_type_conversion_tag_constructor(
        _: yaml.SafeLoader, node: yaml.Node
    ) -> FieldWithMaybeType:
        """Converts a field from one type to another as specified by the tag in the YAML
        file."""
        if not isinstance(node, yaml.ScalarNode):
            raise NotImplementedError
        tag = node.tag[1:].lower()
        if tag == "str":
            new_type = str
        elif tag == "int":
            new_type = int
        elif tag == "float":
            new_type = float
        elif tag == "bool":
            new_type = bool
        else:
            raise NotImplementedError(f"Converting to {tag} is not supported")
        return FieldWithMaybeType(new_type=new_type, name=node.value)

    yaml.SafeLoader.add_constructor("!str", yaml_type_conversion_tag_constructor)
    yaml.SafeLoader.add_constructor("!int", yaml_type_conversion_tag_constructor)
    yaml.SafeLoader.add_constructor("!ref", yaml_reference_tag_constructor)


def resolve_refs(config_dict: dict) -> dict:
    """Resolve FieldReference objects in the config."""
    constants = config_dict.get("constants", {})

    def _resolve(obj):
        if isinstance(obj, FieldReference):
            # Resolve the reference from constants and recursively resolve the result
            resolved = constants.get(obj.reference_name, obj)
            # If we got another reference or complex object, resolve it too
            if resolved != obj:
                return _resolve(resolved)
            return resolved
        elif isinstance(obj, FieldWithMaybeType):
            # Apply type conversion if needed
            value = constants.get(obj.name, obj.name)
            try:
                return obj.new_type(value)
            except (ValueError, TypeError):
                return value
        elif isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        return obj

    # First resolve the entire config
    resolved = _resolve(config_dict)

    # Make sure no FieldReference objects remain anywhere
    def _check_resolved(obj):
        if isinstance(obj, (FieldReference, FieldWithMaybeType)):
            raise ValueError(f"Unresolved reference: {obj}")
        elif isinstance(obj, dict):
            for v in obj.values():
                _check_resolved(v)
        elif isinstance(obj, list):
            for item in obj:
                _check_resolved(item)

    _check_resolved(resolved)
    return resolved


def parse_task_config(task_data) -> TaskConfig:
    """Parse task data into TaskConfig object."""
    if isinstance(task_data, str):
        # Simple task name string
        return TaskConfig(name=task_data)
    elif isinstance(task_data, dict):
        # Task with parameters - use dataclass defaults for missing values
        # Build kwargs dict only with provided values
        kwargs = {"name": task_data["name"]}

        if "task_name" in task_data:
            kwargs["task_name"] = task_data["task_name"]
        if "max_tokens" in task_data:
            kwargs["max_tokens"] = task_data["max_tokens"]
        if "num_fewshot" in task_data:
            kwargs["num_fewshot"] = task_data["num_fewshot"]
        if "temperature" in task_data:
            kwargs["temperature"] = task_data["temperature"]
        if "judge_api_key" in task_data:
            kwargs["judge_api_key"] = task_data["judge_api_key"]
        if "judge_base_url" in task_data:
            kwargs["judge_base_url"] = task_data["judge_base_url"]
        if "judge_name" in task_data:
            kwargs["judge_name"] = task_data["judge_name"]
        if "limit" in task_data:
            kwargs["limit"] = task_data["limit"]

        return TaskConfig(**kwargs)
    else:
        raise ValueError(f"Invalid task data: {task_data}")


def load_samples(output_dir: str, model_name: str, task_name: str) -> list[dict] | None:
    """Load evaluation samples from JSONL files, selecting the newest version of each file."""
    task_output_dir = Path(output_dir) / task_name / model_name.replace("/", "__")

    print(f"Loading samples from: {task_output_dir}")

    # Find all JSONL files
    jsonl_files = list(task_output_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No JSONL files found for {model_name} / {task_name}")
        return None

    # Pattern to extract base name and timestamp from filename
    # Example: samples_lidar_2025-11-17T13-05-01.938804.jsonl
    # Timestamp pattern: YYYY-MM-DDTHH-MM-SS.microseconds
    timestamp_pattern = re.compile(r"(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$")

    # Group files by base name and track the newest version
    file_groups = defaultdict(list)

    for jsonl_file in jsonl_files:
        match = timestamp_pattern.match(jsonl_file.name)
        if match:
            base_name = match.group(1)
            timestamp_str = match.group(2)

            # Parse timestamp (convert '-' to ':' for time part only)
            try:
                # Format: 2025-11-17T13-05-01.938804 -> 2025-11-17T13:05:01.938804
                # Split on 'T' to separate date and time, then replace '-' with ':' only in time part
                date_part, time_part = timestamp_str.split("T")
                time_normalized = time_part.replace("-", ":")
                timestamp_normalized = f"{date_part}T{time_normalized}"
                timestamp = datetime.fromisoformat(timestamp_normalized)
                file_groups[base_name].append((timestamp, jsonl_file))
            except ValueError as e:
                print(f"Warning: Failed to parse timestamp from {jsonl_file.name}: {e}")
                # If timestamp parsing fails, still include the file with a default timestamp
                file_groups[base_name].append((datetime.min, jsonl_file))
        else:
            # If no timestamp pattern found, use the file anyway with a default base name
            file_groups[jsonl_file.stem].append((datetime.min, jsonl_file))

    # Select the newest file for each base name
    selected_files = []
    for base_name, files in file_groups.items():
        # Sort by timestamp (newest first) and select the first one
        files.sort(key=lambda x: x[0], reverse=True)
        newest_file = files[0][1]
        selected_files.append(newest_file)
        if len(files) > 1:
            print(
                f"  Found {len(files)} versions of '{base_name}', using newest: {newest_file.name}"
            )

    # Load all samples from selected files
    all_samples = []
    for sample_file in selected_files:
        try:
            with open(sample_file, "r") as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        all_samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to load samples from {sample_file}: {e}")

    print(f"  Loaded {len(all_samples)} samples from {len(selected_files)} file(s)")
    return all_samples if all_samples else None


def calculate_weighted_stderr(task_stderrs, task_sizes, total_size):
    """
    Calculate weighted standard error for aggregated metric.

    Uses the formula for combining standard errors of independent samples:
    SE_combined = sqrt(sum((n_i/N)^2 * SE_i^2))
    """
    weighted_var = sum(
        ((size / total_size) ** 2) * (stderr**2) for stderr, size in zip(task_stderrs, task_sizes)
    )
    return math.sqrt(weighted_var)


def add_aggregate_metrics_to_results(results_file: Path) -> bool:
    """
    Add aggregate metrics to group tasks if they're missing.

    Calculates weighted averages (by sample size) for group tasks according
    to their aggregate_metric_list configuration.

    Returns True if any metrics were added, False otherwise.
    """
    try:
        # Load results
        with open(results_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Failed to load results file: {e}")
        return False

    # Check if there are any groups
    groups = data.get("groups", {})
    if not groups:
        return False

    modified = False

    # Process each group
    for group_name, group_data in groups.items():
        # Get subtasks for this group
        subtasks = data.get("group_subtasks", {}).get(group_name, [])
        if not subtasks:
            continue

        # Check if aggregate metric is already present
        group_results = data["results"].get(group_name, {})
        if "exact_match,get_response" in group_results:
            # Metric already exists
            continue

        # Calculate aggregate metrics
        task_scores = []
        task_stderrs = []
        task_sizes = []

        for task_name in subtasks:
            # Get exact_match score
            task_result = data["results"].get(task_name, {})
            score = task_result.get("exact_match,get_response")
            stderr = task_result.get("exact_match_stderr,get_response")

            # Get sample size
            n_samples_data = data.get("n-samples", {}).get(task_name, {})
            size = n_samples_data.get("effective", n_samples_data.get("original", 0))

            if score is not None and size > 0:
                task_scores.append(score)
                task_stderrs.append(stderr if stderr is not None else 0)
                task_sizes.append(size)

        if not task_scores:
            continue

        # Calculate weighted average (weight by size)
        total_size = sum(task_sizes)
        weighted_avg = sum(
            score * (size / total_size) for score, size in zip(task_scores, task_sizes)
        )

        # Calculate weighted standard error
        weighted_stderr = calculate_weighted_stderr(task_stderrs, task_sizes, total_size)

        # Update the group results
        if group_name not in data["results"]:
            data["results"][group_name] = {}

        data["results"][group_name]["exact_match,get_response"] = weighted_avg
        data["results"][group_name]["exact_match_stderr,get_response"] = weighted_stderr

        # Also update groups section
        if isinstance(data["groups"][group_name], dict):
            data["groups"][group_name]["exact_match,get_response"] = weighted_avg
            data["groups"][group_name]["exact_match_stderr,get_response"] = weighted_stderr
        else:
            # If groups entry is just a string, convert to dict
            data["groups"][group_name] = {
                "alias": group_name,
                "exact_match,get_response": weighted_avg,
                "exact_match_stderr,get_response": weighted_stderr,
            }

        print(
            f"  ✓ Added aggregate metrics for {group_name}: {weighted_avg:.4f} ± {weighted_stderr:.4f}"
        )
        modified = True

    # Write back to file if modified
    if modified:
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    return modified


def load_eval_results(output_dir: str, model_name: str, task_name: str) -> dict | None:
    """Load evaluation results from lm_eval output directory, selecting the newest file."""
    task_output_dir = Path(output_dir) / task_name / model_name.replace("/", "__")

    print(f"Loading results from: {task_output_dir}")

    # Find results.json file
    results_files = list(task_output_dir.glob("**/*.json"))
    if not results_files:
        print(f"Warning: No json found for {model_name} / {task_name}")
        return None

    # Sort by modification time (newest first) to ensure we get the most recent results
    results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    results_file = results_files[0]

    if len(results_files) > 1:
        print(f"  Found {len(results_files)} result files, using newest: {results_file.name}")

    with open(results_file, "r") as f:
        data = json.load(f)

    return data.get("results", {})


def flatten_metrics(results: dict, original_task_name: str) -> dict:
    """Flatten nested metrics dictionary for wandb logging."""
    flattened = {}

    for task_name, metrics in results.items():
        # Skip 'alias' and 'group' keys
        for key, value in metrics.items():
            if key in ["alias", "group"]:
                continue

            # Create hierarchical metric name: original_task/subtask/metric
            # Remove any commas from metric names (lm_eval uses commas for variants)
            clean_key = key.split(",")[0] if "," in key else key

            # If task_name is different from original_task_name, it's a subtask
            if task_name != original_task_name:
                metric_name = f"{original_task_name}/{task_name}/{clean_key}"
            else:
                metric_name = f"{original_task_name}/{clean_key}"

            # Only log numeric values
            if isinstance(value, (int, float)):
                flattened[metric_name] = value

    return flattened


def init_wandb_for_task(wandb_config: WandbConfig, model: ModelConfig, task: TaskConfig):
    """Initialize wandb run for a specific model and task."""
    if not wandb_config.enabled:
        return None

    try:
        # Login to wandb with API key if provided (only once)
        if wandb_config.api_key and not wandb.run:
            wandb.login(key=wandb_config.api_key)

        # Create run name: prefix_model_task
        model_short_name = model.name.split("/")[-1]
        run_name_parts = []
        if wandb_config.run_name:
            run_name_parts.append(wandb_config.run_name)
        run_name_parts.extend([model_short_name, task.name])
        run_name = "_".join(run_name_parts)

        # Initialize wandb run
        run_config = {
            "model_name": model.name,
            "task_name": task.name,  # User-defined task configuration name
            "lm_eval_task": task.task_name,  # Actual lm_eval task name
            "temperature": model.temperature,
            "max_tokens": task.max_tokens,
            "num_fewshot": task.num_fewshot,
        }

        wandb_init_kwargs = {
            "project": wandb_config.project,
            "name": run_name,
            "config": run_config,
        }

        if wandb_config.entity:
            wandb_init_kwargs["entity"] = wandb_config.entity

        if wandb_config.tags:
            wandb_init_kwargs["tags"] = wandb_config.tags

        run = wandb.init(**wandb_init_kwargs)
        return run

    except Exception as e:
        print(f"\n✗ Failed to initialize W&B: {e}")
        print("Continuing without W&B logging...")
        return None


def create_samples_table(samples: list[dict]) -> wandb.Table:
    """Create a wandb table from evaluation samples, dynamically handling all fields."""
    if not samples:
        return wandb.Table(columns=["empty"], data=[])

    # First pass: collect all unique column names across all samples
    all_columns = set()

    for sample in samples:
        # Add top-level fields
        all_columns.update(sample.keys())

        # Add doc fields with 'doc.' prefix
        doc = sample.get("doc", {})
        if isinstance(doc, dict):
            for key in doc.keys():
                all_columns.add(f"doc.{key}")

    # Sort columns for consistent ordering, with important fields first
    priority_fields = ["doc_id", "doc.Topic", "doc.Question", "target", "filtered_resps"]
    columns = []

    # Add priority fields first if they exist
    for field in priority_fields:
        if field in all_columns:
            columns.append(field)
            all_columns.remove(field)

    # Add remaining fields sorted alphabetically
    columns.extend(sorted(all_columns))

    # Create table
    table = wandb.Table(columns=columns)

    # Populate table
    for sample in samples:
        row = []
        doc = sample.get("doc", {})

        for col in columns:
            if col.startswith("doc."):
                # Handle doc fields
                field_name = col[4:]  # Remove 'doc.' prefix
                value = doc.get(field_name, "")
            else:
                # Handle top-level fields
                value = sample.get(col, "")

            # Convert value to string representation for table
            if value is None:
                row.append("")
            elif isinstance(value, list):
                # For lists, join elements or show first element
                if col == "filtered_resps" and value:
                    # For model responses, take first one
                    row.append(str(value[0]))
                else:
                    # For other lists, show as JSON
                    row.append(json.dumps(value))
            elif isinstance(value, dict):
                # For dicts, show as JSON
                row.append(json.dumps(value))
            else:
                row.append(str(value))

        table.add_data(*row)

    return table


def log_task_metrics_to_wandb(task_metrics: dict, model: ModelConfig, task: TaskConfig):
    """Log metrics for a task to the active wandb run."""
    # Create flat metrics dict (without model name prefix since each run is per model/task)
    flat_metrics = {}
    summary_table = wandb.Table(columns=["model_name", "task", "subtask", "metric", "value"])

    for metric_name, value in sorted(task_metrics.items()):
        # Split metric_name (format: task/metric or task/subtask/metric)
        parts = metric_name.split("/")

        if len(parts) == 2:
            # Format: task/metric - just use the metric name
            _, metric = parts
            flat_metrics[metric] = value
            summary_table.add_data(model.name, task.name, "", metric, value)
        elif len(parts) == 3:
            # Format: task/subtask/metric - use subtask/metric
            _, subtask, metric = parts
            metric_key = f"{subtask}/{metric}"
            flat_metrics[metric_key] = value
            # Add subtask in separate column
            summary_table.add_data(model.name, task.name, subtask, metric, value)
        else:
            # Fallback for unexpected format
            flat_metrics[metric_name] = value
            summary_table.add_data(model.name, task.name, "", metric_name, value)

    # Log metrics
    wandb.log(flat_metrics)
    wandb.log({"metrics_summary": summary_table})


def run_evaluation(model: ModelConfig, task: TaskConfig, output_dir: str, hf_token: str | None = None) -> int:
    """Run lm_eval for a specific model and task."""

    # Construct model_args
    model_args_parts = [
        f"base_url={model.base_url}",
        f"model={model.name}",
        f"num_concurrent={model.num_concurrent}",
        f"max_tokens={task.max_tokens}",
        f"temperature={task.temperature if task.temperature > 0 else model.temperature}",
        f"timeout={model.timeout}",
    ]
    model_args = ",".join(model_args_parts)

    # Create output directory for this specific evaluation (use user-defined name)
    task_output_dir = Path(output_dir) / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Build the command as a list (safer than shell=True)
    # Use task_name (actual lm_eval task) for the --tasks parameter
    eval_command = [
        "lm_eval",
        "--model",
        "openai-chat-completions",
        "--model_args",
        model_args,
        "--include",
        "tasks",
        "--tasks",
        task.task_name,  # Use task_name for the actual lm_eval task
        "--num_fewshot",
        str(task.num_fewshot),
        "--output_path",
        str(task_output_dir),
        "--log_samples",
        "--apply_chat_template",
    ]
    # Add limit if specified
    if task.limit is not None and task.limit > 0:
        eval_command.extend(["--limit", str(task.limit)])

    # Set API key as environment variable (more secure than command line)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = model.api_key

    # Set HuggingFace token if provided in config, otherwise use environment variable
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # If not in config, check if it exists in environment and preserve it
    elif "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" in os.environ:
        env["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HUGGING_FACE_HUB_TOKEN" not in env and "HF_TOKEN" in os.environ:
        env["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # Set judge environment variables if provided
    if task.judge_api_key:
        env["JUDGE_API_KEY"] = task.judge_api_key
    if task.judge_base_url:
        env["JUDGE_BASE_URL"] = task.judge_base_url
    if task.judge_name:
        env["JUDGE_NAME"] = task.judge_name

    print(f"\nRunning evaluation:")
    print(f"  Model: {model.name}")
    print(f"  Task Config: {task.name}")
    if task.name != task.task_name:
        print(f"  LM-Eval Task: {task.task_name}")
    print(f"  Output: {task_output_dir}")
    print(f"  Command: {' '.join(eval_command)}")
    print("-" * 80)

    try:
        result = subprocess.run(eval_command, env=env, check=True, capture_output=False)
        print(f"✓ Completed: {model.name} on {task.name}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {model.name} on {task.name}")
        print(f"  Error: {e}")
        return e.returncode
    except FileNotFoundError:
        print("Error: lm_eval command not found.")
        print("Install with: pip install lm-eval")
        return 1


def evaluate_model(model: ModelConfig, output_dir: str, hf_token: str | None = None) -> dict[str, int]:
    """Evaluate a model on all its tasks."""
    results = {}

    for task in model.tasks:
        return_code = run_evaluation(model, task, output_dir, hf_token)
        results[task.name] = return_code

    return results


def main(config_file: str):
    """Main entry point for the evaluation script."""

    # Register custom YAML tags
    _register_yaml_tags()

    # Load and parse config
    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Resolve !ref references
    config_dict = resolve_refs(config_dict)

    # Parse wandb config if present
    wandb_config = WandbConfig()
    if "wandb" in config_dict:
        wandb_data = config_dict["wandb"]
        wandb_config = WandbConfig(
            enabled=wandb_data.get("enabled", False),
            project=wandb_data.get("project", "eve-evaluation"),
            entity=wandb_data.get("entity"),
            run_name=wandb_data.get("run_name"),
            tags=wandb_data.get("tags", []),
            api_key=wandb_data.get("api_key"),
        )

    # Parse into dataclass structure
    eval_config = EvaluationConfig(
        models=[
            ModelConfig(
                name=model["name"],
                base_url=model["base_url"],
                api_key=model.get("api_key", ""),
                temperature=model.get("temperature", 0.0),
                num_concurrent=model.get("num_concurrent", 3),
                timeout=model.get("timeout", 300),
                tasks=[parse_task_config(task) for task in model["tasks"]],
            )
            for model in config_dict["models"]
        ],
        output_dir=config_dict.get("output_dir", "eval_results"),
        wandb=wandb_config,
        hf_token=config_dict.get("constants", {}).get("hf_token"),
    )

    # Create output directory
    Path(eval_config.output_dir).mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}

    # Evaluate each model
    for model in eval_config.models:
        print(f"\n{'=' * 80}")
        print(f"Evaluating model: {model.name}")
        print(f"Tasks: {[task.name for task in model.tasks]}")
        print(f"{'=' * 80}")

        model_results = evaluate_model(model, eval_config.output_dir, eval_config.hf_token)
        all_results[model.name] = model_results

        # Add aggregate metrics for group tasks if missing (do this for all tasks)
        for task in model.tasks:
            task_return_code = model_results.get(task.name, 1)
            if task_return_code != 0:
                continue  # Skip failed tasks

            print(f"\nChecking aggregate metrics for {task.name}...")
            task_output_dir = (
                Path(eval_config.output_dir)
                / task.name
                / model.name.replace("/", "__")
            )
            results_files = list(task_output_dir.glob("**/*.json"))
            if results_files:
                # Use the newest results file
                results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                results_file = results_files[0]
                add_aggregate_metrics_to_results(results_file)

        # Log each task separately to wandb if enabled
        if eval_config.wandb.enabled:
            for task in model.tasks:
                # Check if the task evaluation succeeded before logging to wandb
                task_return_code = model_results.get(task.name, 1)
                if task_return_code != 0:
                    print(f"\nSkipping W&B logging for {task.name} (evaluation failed)")
                    continue

                print(f"\nLoading evaluation results for W&B logging ({task.name})...")
                task_results = load_eval_results(eval_config.output_dir, model.name, task.name)

                if task_results:
                    # Initialize a new wandb run for this model/task combination
                    wandb_run = init_wandb_for_task(eval_config.wandb, model, task)

                    if wandb_run:
                        # Flatten and log metrics for this task
                        task_metrics = flatten_metrics(task_results, task.name)
                        log_task_metrics_to_wandb(task_metrics, model, task)

                        # Load and log samples
                        print(f"Loading samples for W&B logging ({task.name})...")
                        samples = load_samples(eval_config.output_dir, model.name, task.name)

                        if samples:
                            samples_table = create_samples_table(samples)
                            wandb.log({"samples": samples_table})
                            print(f"✓ Logged {len(samples)} samples to W&B")
                        else:
                            print(f"Warning: No samples found for {task.name}")

                        print(f"✓ Logged {len(task_metrics)} metrics to W&B: {wandb_run.url}")

                        # Finish this run
                        wandb.finish()
                else:
                    print(f"Warning: No metrics found for {task.name}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")

    total_evaluations = 0
    failed_evaluations = 0

    for model_name, tasks_results in all_results.items():
        print(f"\n{model_name}:")
        for task_name, return_code in tasks_results.items():
            status = "✓ PASSED" if return_code == 0 else "✗ FAILED"
            print(f"  {task_name}: {status}")
            total_evaluations += 1
            if return_code != 0:
                failed_evaluations += 1

    print(f"\n{'=' * 80}")
    print(f"Total: {total_evaluations} evaluations")
    print(f"Passed: {total_evaluations - failed_evaluations}")
    print(f"Failed: {failed_evaluations}")
    print(f"{'=' * 80}\n")

    return 0 if failed_evaluations == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run lm_eval harness based on a YAML config file")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")

    args = parser.parse_args()

    exit_code = main(args.config)
    exit(exit_code)
