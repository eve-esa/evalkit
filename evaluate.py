import argparse
import dataclasses
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml


@dataclasses.dataclass
class FieldReference:
    reference_name: str


@dataclasses.dataclass
class FieldWithMaybeType:
    new_type: type
    name: str


@dataclasses.dataclass
class TaskConfig:
    name: str  # User-defined name for this task configuration (used for output folders)
    task_name: str | None = None  # Actual lm_eval task name (defaults to name if not provided)
    model_type: str = "local-chat-completions"
    max_tokens: int = 512
    num_fewshot: int = 0
    temperature: float = 0.0
    apply_chat_template: bool = True  # Whether to apply chat template to prompts
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_name: str = ""
    batch_size: int = 1
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
    tokenizer: str | None = None  # HuggingFace tokenizer name for chat template (optional)


@dataclasses.dataclass
class EvaluationConfig:
    models: list[ModelConfig]
    output_dir: str = "eval_results"
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
        if "model_type" in task_data:
            kwargs["model_type"] = task_data["model_type"]
        if "max_tokens" in task_data:
            kwargs["max_tokens"] = task_data["max_tokens"]
        if "num_fewshot" in task_data:
            kwargs["num_fewshot"] = task_data["num_fewshot"]
        if "temperature" in task_data:
            kwargs["temperature"] = task_data["temperature"]
        if "apply_chat_template" in task_data:
            kwargs["apply_chat_template"] = task_data["apply_chat_template"]
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


def run_evaluation(
    model: ModelConfig, task: TaskConfig, output_dir: str, hf_token: str | None = None
) -> tuple[int, dict | None]:
    """Run lm_eval for a specific model and task using simple_evaluate.

    Returns:
        tuple: (return_code, results_dict) where results_dict is None if evaluation failed
    """
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager

    # Reset judge state to pick up new environment variables for this task
    try:
        from metrics.judge_utils import reset_judge_state

        reset_judge_state()
    except ImportError:
        pass  # judge_utils not available, skip reset

    # Create output directory for this specific evaluation (use user-defined name)
    task_output_dir = Path(output_dir) / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables before initializing the model
    env_backup = {}

    # Set API key as environment variable (more secure than command line)
    env_backup["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = model.api_key

    # Set HuggingFace token if provided in config, otherwise use environment variable
    if hf_token:
        env_backup["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        env_backup["HUGGING_FACE_HUB_TOKEN"] = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # If not in config, check if it exists in environment and preserve it
    elif "HUGGING_FACE_HUB_TOKEN" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
    elif "HF_TOKEN" in os.environ and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # Set judge environment variables if provided
    if task.judge_api_key:
        env_backup["JUDGE_API_KEY"] = os.environ.get("JUDGE_API_KEY")
        os.environ["JUDGE_API_KEY"] = task.judge_api_key
    if task.judge_base_url:
        env_backup["JUDGE_BASE_URL"] = os.environ.get("JUDGE_BASE_URL")
        os.environ["JUDGE_BASE_URL"] = task.judge_base_url
    if task.judge_name:
        env_backup["JUDGE_NAME"] = os.environ.get("JUDGE_NAME")
        os.environ["JUDGE_NAME"] = task.judge_name

    print("\nRunning evaluation:")
    print(f"  Model: {model.name}")
    print(f"  Model Type: {task.model_type}")
    print(f"  Task Config: {task.name}")
    if task.name != task.task_name:
        print(f"  LM-Eval Task: {task.task_name}")
    print(f"  Output: {task_output_dir}")
    print("-" * 80)

    try:
        # Construct full API URL based on model type
        # Base URL should be just the base (e.g., https://api.openai.com/v1/)
        # We append either 'completions' or 'chat/completions' based on model type
        base_url = model.base_url.rstrip("/")  # Remove trailing slashes

        # Check if "chat" is in the model type to determine the endpoint
        if "chat" in task.model_type.lower():
            full_url = f"{base_url}/chat/completions"
        else:
            full_url = f"{base_url}/completions"

        # Construct model arguments as a string (lm_eval expects comma-separated key=value pairs)
        model_args = (
            f"base_url={full_url},"
            f"model={model.name},"
            f"num_concurrent={model.num_concurrent},"
            f"max_tokens={task.max_tokens},"
            f"temperature={task.temperature if task.temperature > 0 else model.temperature},"
            f"timeout={model.timeout},"
            f"batch_size={task.batch_size}"
        )

        # Add tokenizer if provided
        if model.tokenizer:
            model_args += f",tokenizer={model.tokenizer}"

        # Force huggingface tokenizer backend and send prompts as strings
        # to avoid vLLM MistralTokenizer batch_decode issues
        model_args += ",tokenizer_backend=huggingface,tokenized_requests=False"

        # Create TaskManager to include custom tasks directory
        # This is equivalent to the CLI's --include tasks option
        task_manager = TaskManager(include_path="tasks")

        # Run evaluation using simple_evaluate
        # Pass model type as string and let simple_evaluate handle model initialization
        results = simple_evaluate(
            model=task.model_type,
            model_args=model_args,
            tasks=[task.task_name],
            num_fewshot=task.num_fewshot,
            limit=task.limit if task.limit and task.limit > 0 else None,
            log_samples=True,
            apply_chat_template=task.apply_chat_template,
            task_manager=task_manager,
        )

        if results is None:
            raise RuntimeError("simple_evaluate returned None")

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")

        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        # Save samples as JSONL files
        if "samples" in results:
            for subtask_name, samples_list in results["samples"].items():
                samples_path = task_output_dir / f"samples_{subtask_name}_{timestamp}.jsonl"
                print(f"  Saving samples to: {samples_path}")
                with open(samples_path, "w") as f:
                    for sample in samples_list:
                        f.write(json.dumps(sample, default=default_serializer) + "\n")

        # Save results without samples (samples are in JSONL files)
        results_to_save = results.copy()
        if "samples" in results_to_save:
            del results_to_save["samples"]

        results_path = task_output_dir / f"results_{timestamp}.json"
        print(f"  Saving results to: {results_path}")
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=2, default=default_serializer)

        print(f"✓ Completed: {model.name} on {task.name}")
        return_code = 0
        eval_results = results

    except Exception as e:
        print(f"✗ Failed: {model.name} on {task.name}")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return_code = 1
        eval_results = None

    finally:
        # Restore environment variables
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    return return_code, eval_results


def evaluate_model(
    model: ModelConfig, output_dir: str, hf_token: str | None = None
) -> dict[str, int]:
    """Evaluate a model on all its tasks."""
    results = {}

    for task in model.tasks:
        return_code, eval_results = run_evaluation(model, task, output_dir, hf_token)
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
                tokenizer=model.get("tokenizer"),
                tasks=[parse_task_config(task) for task in model["tasks"]],
            )
            for model in config_dict["models"]
        ],
        output_dir=config_dict.get("output_dir", "eval_results"),
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
