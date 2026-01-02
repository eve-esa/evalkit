import argparse
import dataclasses
import json
import math
import os
import re
import traceback
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
    name: str
    task_name: str | None = None
    max_tokens: int = 512
    num_fewshot: int = 0
    temperature: float = 0.0
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_name: str = ""
    limit: int | None = None

    def __post_init__(self):
        if self.task_name is None:
            self.task_name = self.name


@dataclasses.dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    tasks: list[TaskConfig]
    temperature: float = 0.0
    num_concurrent: int = 3
    tokenizer: str | None = None
    timeout: int = 300
    custom_task_dir: str | None = None


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
    constants = config_dict.get("constants", {})

    def _resolve(obj):
        if isinstance(obj, FieldReference):
            resolved = constants.get(obj.reference_name, obj)
            if resolved != obj:
                return _resolve(resolved)
            return resolved
        elif isinstance(obj, FieldWithMaybeType):
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

    resolved = _resolve(config_dict)

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
    if isinstance(task_data, str):
        return TaskConfig(name=task_data)
    elif isinstance(task_data, dict):
        kwargs = {"name": task_data["name"]}
        for field in ["task_name", "max_tokens", "num_fewshot", "temperature",
                      "judge_api_key", "judge_base_url", "judge_name", "limit"]:
            if field in task_data:
                kwargs[field] = task_data[field]
        return TaskConfig(**kwargs)
    else:
        raise ValueError(f"Invalid task data: {task_data}")


def load_samples(output_dir: str, model_name: str, task_name: str) -> list[dict] | None:
    task_output_dir = Path(output_dir) / task_name / model_name.replace("/", "__")
    print(f"Loading samples from: {task_output_dir}")
    jsonl_files = list(task_output_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        print(f"Warning: No JSONL files found for {model_name} / {task_name}")
        return None

    timestamp_pattern = re.compile(r"(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl$")
    file_groups = defaultdict(list)

    for jsonl_file in jsonl_files:
        match = timestamp_pattern.match(jsonl_file.name)
        if match:
            base_name = match.group(1)
            timestamp_str = match.group(2)
            try:
                date_part, time_part = timestamp_str.split("T")
                time_normalized = time_part.replace("-", ":")
                timestamp_normalized = f"{date_part}T{time_normalized}"
                timestamp = datetime.fromisoformat(timestamp_normalized)
                file_groups[base_name].append((timestamp, jsonl_file))
            except ValueError:
                file_groups[base_name].append((datetime.min, jsonl_file))
        else:
            file_groups[jsonl_file.stem].append((datetime.min, jsonl_file))

    selected_files = []
    for base_name, files in file_groups.items():
        files.sort(key=lambda x: x[0], reverse=True)
        selected_files.append(files[0][1])

    all_samples = []
    for sample_file in selected_files:
        try:
            with open(sample_file, "r") as f:
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Failed to load samples from {sample_file}: {e}")

    return all_samples if all_samples else None


def calculate_weighted_stderr(task_stderrs, task_sizes, total_size):
    weighted_var = sum(
        ((size / total_size) ** 2) * (stderr**2) for stderr, size in zip(task_stderrs, task_sizes)
    )
    return math.sqrt(weighted_var)


def add_aggregate_metrics_to_results(results_file: Path) -> bool:
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Failed to load results file: {e}")
        return False

    groups = data.get("groups", {})
    if not groups:
        return False

    modified = False
    for group_name, group_data in groups.items():
        subtasks = data.get("group_subtasks", {}).get(group_name, [])
        if not subtasks:
            continue
        group_results = data["results"].get(group_name, {})
        if "exact_match,get_response" in group_results:
            continue

        task_scores = []
        task_stderrs = []
        task_sizes = []

        for task_name in subtasks:
            task_result = data["results"].get(task_name, {})
            score = task_result.get("exact_match,get_response")
            stderr = task_result.get("exact_match_stderr,get_response")
            n_samples_data = data.get("n-samples", {}).get(task_name, {})
            size = n_samples_data.get("effective", n_samples_data.get("original", 0))

            if score is not None and size > 0:
                task_scores.append(score)
                task_stderrs.append(stderr if stderr is not None else 0)
                task_sizes.append(size)

        if not task_scores:
            continue

        total_size = sum(task_sizes)
        weighted_avg = sum(
            score * (size / total_size) for score, size in zip(task_scores, task_sizes)
        )
        weighted_stderr = calculate_weighted_stderr(task_stderrs, task_sizes, total_size)

        if group_name not in data["results"]:
            data["results"][group_name] = {}
        data["results"][group_name]["exact_match,get_response"] = weighted_avg
        data["results"][group_name]["exact_match_stderr,get_response"] = weighted_stderr

        if isinstance(data["groups"][group_name], dict):
            data["groups"][group_name]["exact_match,get_response"] = weighted_avg
            data["groups"][group_name]["exact_match_stderr,get_response"] = weighted_stderr
        else:
            data["groups"][group_name] = {
                "alias": group_name,
                "exact_match,get_response": weighted_avg,
                "exact_match_stderr,get_response": weighted_stderr,
            }
        modified = True

    if modified:
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
    return modified


def load_eval_results(output_dir: str, model_name: str, task_name: str) -> dict | None:
    task_output_dir = Path(output_dir) / task_name / model_name.replace("/", "__")
    print(f"Loading results from: {task_output_dir}")
    results_files = list(task_output_dir.glob("**/*.json"))
    if not results_files:
        print(f"Warning: No json found for {model_name} / {task_name}")
        return None
    results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    results_file = results_files[0]
    with open(results_file, "r") as f:
        data = json.load(f)
    return data.get("results", {})


def run_evaluation(model: ModelConfig, task: TaskConfig, output_dir: str, hf_token: str | None = None) -> int:
    """Run lm_eval using simple_evaluate API with local-completions."""
    
    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager

    os.environ["OPENAI_API_KEY"] = model.api_key
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    
    if task.judge_api_key:
        os.environ["JUDGE_API_KEY"] = task.judge_api_key
    if task.judge_base_url:
        os.environ["JUDGE_BASE_URL"] = task.judge_base_url
    if task.judge_name:
        os.environ["JUDGE_NAME"] = task.judge_name

    custom_tasks_path = model.custom_task_dir
    task_manager = TaskManager(include_path=custom_tasks_path)

    model_args = {
        "base_url": model.base_url,
        "model": model.name,
        "num_concurrent": model.num_concurrent,
        "max_tokens": task.max_tokens,
        "temperature": task.temperature if task.temperature > 0 else model.temperature,
        "timeout": model.timeout,
    }
    
    if model.tokenizer:
        model_args["tokenizer"] = model.tokenizer

    model_name_sanitized = model.name.replace("/", "__")
    task_output_dir = Path(output_dir) / task.name / model_name_sanitized
    task_output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning evaluation:")
    print(f"  Model: {model.name}")
    print(f"  Tokenizer: {model_args.get('tokenizer', 'Using model path (Default)')}")
    print(f"  Task Config: {task.name}")
    if task.name != task.task_name:
        print(f"  LM-Eval Task: {task.task_name}")
    print(f"  Output: {task_output_dir}")
    print("-" * 80)

    try:
        results = simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=[task.task_name],
            task_manager=task_manager,
            num_fewshot=task.num_fewshot,
            limit=task.limit,
            log_samples=True,
        )

        if results is None:
            raise RuntimeError("simple_evaluate returned None")

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        results_path = task_output_dir / f"results_{timestamp}.json"
        
        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)
            return str(obj)

        print(f"  Saving results to: {results_path}")
        if "samples" in results:
            for subtask_name, samples_list in results["samples"].items():
                samples_path = task_output_dir / f"samples_{subtask_name}_{timestamp}.jsonl"
                print(f"  Saving samples to: {samples_path}")
                with open(samples_path, "w") as f:
                    for sample in samples_list:
                        f.write(json.dumps(sample, default=default_serializer) + "\n")

        results_to_save = results.copy()
        if "samples" in results_to_save:
            del results_to_save["samples"]

        results_path = task_output_dir / f"results_{timestamp}.json"
        print(f"  Saving results to: {results_path}")
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=2, default=default_serializer)

        print(f"✓ Completed: {model.name} on {task.name}")
        return 0

    except Exception as e:
        print(f"✗ Failed: {model.name} on {task.name}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return 1


def evaluate_model(
    model: ModelConfig, output_dir: str, hf_token: str | None = None
) -> dict[str, int]:
    """Evaluate a model on all its tasks."""
    results = {}

    for task in model.tasks:
        return_code = run_evaluation(model, task, output_dir, hf_token)
        results[task.name] = return_code

        if return_code == 0:
            print(f"\nChecking aggregate metrics for {task.name}...")
            task_output_dir = Path(output_dir) / task.name / model.name.replace("/", "__")
            results_files = list(task_output_dir.glob("**/*.json"))
            if results_files:
                results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                results_file = results_files[0]
                add_aggregate_metrics_to_results(results_file)

    return results


def main(config_file: str):
    _register_yaml_tags()

    with open(config_file, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    config_dict = resolve_refs(config_dict)

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
                tokenizer=model["tokenizer"],
                custom_task_dir=model.get("custom_task_dir",None)

            )
            for model in config_dict["models"]
        ],
        output_dir=config_dict.get("output_dir", "eval_results"),
        hf_token=config_dict.get("constants", {}).get("hf_token"),
    )

    Path(eval_config.output_dir).mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model in eval_config.models:
        print(f"\n{'=' * 80}")
        print(f"Evaluating model: {model.name}")
        print(f"Tasks: {[task.name for task in model.tasks]}")
        print(f"{'=' * 80}")

        model_results = evaluate_model(
            model, eval_config.output_dir, eval_config.hf_token
        )
        all_results[model.name] = model_results

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