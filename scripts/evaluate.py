import argparse
import dataclasses
import os
import subprocess
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
    max_tokens: int = 512
    num_fewshot: int = 0
    temperature: float = 0.0
    judge_api_key: str = ""
    judge_base_url: str = ""
    judge_name: str = ""
    limit: int | None = None  # Limit number of samples to evaluate


@dataclasses.dataclass
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    tasks: list[TaskConfig]
    temperature: float = 0.0


@dataclasses.dataclass
class EvaluationConfig:
    models: list[ModelConfig]
    output_dir: str = "eval_results"


def _register_yaml_tags():
    def yaml_reference_tag_constructor(_: yaml.SafeLoader, node: yaml.Node) -> FieldReference:
        if not isinstance(node, yaml.ScalarNode):
            raise NotImplementedError
        if node.tag != "!ref":
            raise NotImplementedError(f"Tag {node.tag} is not supported")
        return FieldReference(reference_name=node.value)

    def yaml_type_conversion_tag_constructor(_: yaml.SafeLoader, node: yaml.Node) -> FieldWithMaybeType:
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
    constants = config_dict.get('constants', {})

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
        # Task with parameters
        return TaskConfig(
            name=task_data['name'],
            max_tokens=task_data.get('max_tokens', 512),
            num_fewshot=task_data.get('num_fewshot', 0),
            temperature=task_data.get('temperature', 0.0),
            judge_api_key=task_data.get('judge_api_key', ''),
            judge_base_url=task_data.get('judge_base_url', ''),
            judge_name=task_data.get('judge_name', ''),
            limit=task_data.get('limit', None)
        )
    else:
        raise ValueError(f"Invalid task data: {task_data}")


def run_evaluation(model: ModelConfig, task: TaskConfig, output_dir: str) -> int:
    """Run lm_eval for a specific model and task."""

    # Construct model_args
    model_args_parts = [
        f"base_url={model.base_url}",
        f"model={model.name}",
        f"num_concurrent=10",
        f"max_tokens={task.max_tokens}",
        f"temperature={task.temperature if task.temperature > 0 else model.temperature}",
    ]
    model_args = ",".join(model_args_parts)

    # Create output directory for this specific evaluation
    task_output_dir = Path(output_dir) / model.name.replace('/', '_') / task.name
    task_output_dir.mkdir(parents=True, exist_ok=True)


    # Build the command as a list (safer than shell=True)
    eval_command = [
        "lm_eval",
        "--model", "openai-chat-completions",
        "--model_args", model_args,
        "--include", "tasks",
        "--tasks", task.name,
        "--num_fewshot", str(task.num_fewshot),
        "--output_path", str(task_output_dir),
        "--log_samples",
        "--apply_chat_template"
    ]
    # Add limit if specified
    if task.limit is not None and task.limit > 0:
        eval_command.extend(["--limit", str(task.limit)])

    # Set API key as environment variable (more secure than command line)
    env = os.environ.copy()
    env['OPENAI_API_KEY'] = model.api_key

    # Set judge environment variables if provided
    if task.judge_api_key:
        env['JUDGE_API_KEY'] = task.judge_api_key
    if task.judge_base_url:
        env['JUDGE_BASE_URL'] = task.judge_base_url
    if task.judge_name:
        env['JUDGE_NAME'] = task.judge_name

    print(f"\nRunning evaluation:")
    print(f"  Model: {model.name}")
    print(f"  Task: {task.name}")
    print(f"  Output: {task_output_dir}")
    print(f"  Command: {' '.join(eval_command)}")
    print("-" * 80)

    try:
        result = subprocess.run(
            eval_command,
            env=env,
            check=True,
            capture_output=False
        )
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


def evaluate_model(model: ModelConfig, output_dir: str) -> dict[str, int]:
    """Evaluate a model on all its tasks."""
    results = {}

    for task in model.tasks:
        return_code = run_evaluation(model, task, output_dir)
        results[task.name] = return_code

    return results


def main(config_file: str):
    """Main entry point for the evaluation script."""

    # Register custom YAML tags
    _register_yaml_tags()

    # Load and parse config
    with open(config_file, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Resolve !ref references
    config_dict = resolve_refs(config_dict)

    # Parse into dataclass structure
    eval_config = EvaluationConfig(
        models=[
            ModelConfig(
                name=model['name'],
                base_url=model['base_url'],
                api_key=model.get('api_key', ''),
                temperature=model.get('temperature', 0.0),
                tasks=[parse_task_config(task) for task in model['tasks']]
            ) for model in config_dict['models']
        ],
        output_dir=config_dict.get('output_dir', 'eval_results')
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

        model_results = evaluate_model(model, eval_config.output_dir)
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
    parser = argparse.ArgumentParser(
        description="Run lm_eval harness based on a YAML config file"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file"
    )

    args = parser.parse_args()

    exit_code = main(args.config)
    exit(exit_code)