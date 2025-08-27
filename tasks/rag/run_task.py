from typing import Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, create_model, field_validator, model_validator

from datasets import load_dataset
from model import ApiModel, _convert_schema_to_model
import yaml
import argparse

from loguru import logger
import json



class ModelConfiguration(BaseModel):
    model_name: str
    temperature: float = 1.0
    max_tokens: int = 1024
    prompt: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    response_model: Optional[Dict[str, Any]] = None
    prompt_response_schema_key: str = "response_schema_json"
    batch_size: Optional[int] = 2
    @field_validator("response_model", mode="before")
    @classmethod
    def _ensure_dict(cls, v):
        if v is not None and not isinstance(v, dict):
            raise ValueError("response_model must be a dictionary (optionally nested)")
        return v

    @field_validator("response_model", mode="before")
    @classmethod
    def _ensure_can_create_model(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is None:
            return v
        _ = _convert_schema_to_model("ResponseModel", v)
        return v



class TaskConfiguration(BaseModel):
    model: ModelConfiguration
    judge: ModelConfiguration
    task_name: str
    dataset: str
    split: Optional[str]  = None
    subset: Optional[str] = None



def tmp_eval(preds: list[list[int]], refs: list[list[int]]):
    acc = []
    iou = []
    for pred, ref in zip(preds, refs):
        if not isinstance(pred, list):
            raise ValueError(f"Expected a list of integers, got {type(pred)} instead.")
        # Compute IoU accuracy and strict matching
        print(pred)
        intersection = len(set(pred) & set(ref))
        union = len(set(pred) | set(ref))
        if union == 0:
            iou.append(0.0)
        else:
            iou.append(intersection / union)
        acc.append(set(pred) == set(ref))
    iou = sum(iou) / len(iou) if iou else 0.0
    acc = sum(acc) / len(acc) if acc else 0.0
    logger.info(f"Evaluation results: IoU: {iou}, Accuracy: {acc}")
    return iou, acc






def run_task(task_config: TaskConfiguration, limit=None) -> Dict[str, Any]:
    """
    Run the specified task with the given configuration and input data.

    :param task_config: Configuration for the task including model and dataset details.
    :param input_data: Input data for the task.
    :return: Output data after processing.
    """
    # Here you would implement the logic to run the task using the model configuration
    # and process the input data accordingly. This is a placeholder for demonstration.

    inputs = []
    if task_config.dataset.endswith(".jsonl"):
        with open(task_config.dataset, "r") as f:
            for line in f:
                inputs.append(json.loads(line))
    else:
        dataset = load_dataset(task_config.dataset, task_config.subset, split=task_config.split)
        inputs = dataset.to_list()

    client = ApiModel(**task_config.model.model_dump())

    if limit is not None:
        inputs = inputs[:limit]


    judge = ApiModel(**task_config.judge.model_dump())

    logger.info(f"Generation started...")
    outputs = client.process_batch(inputs)
    print(outputs)

    iou, acc = tmp_eval([output['citations'] for output in outputs], [single_input['ids'] for single_input in inputs])

    print(f"IoU: {iou}")
    print(f"Accuracy: {acc}")

    outputs = [{f'model_{k}': v for k, v in output.items()} for output in outputs]
    for single_input, single_output in zip(inputs, outputs):
         single_input.update(single_output)
    logger.info(f"Evaluation started...")
    judge_outputs = judge.process_batch(inputs)

    correct_answers = []
    wrong_answers = []
    for i, judge_output in enumerate(judge_outputs):
        if judge_output['correct']:
            correct_answers.append(i)
        else:
            wrong_answers.append(i)

    iou_correct, acc_correct = tmp_eval([outputs[i]['model_citations'] for i in correct_answers], [inputs[i]['ids'] for i in correct_answers])
    iou_wrong, acc_wrong = tmp_eval([outputs[i]['model_citations'] for i in wrong_answers], [inputs[i]['ids'] for i in wrong_answers])
    logger.info(f"IoU: {iou_correct}, Accuracy: {acc_correct}")
    logger.info(f"IoU (wrong): {iou_wrong}, Accuracy (wrong): {acc_wrong}")
    return judge_outputs

if __name__ == "__main__":
    # Read input path from cli
    parser = argparse.ArgumentParser(description="Run RAG task with specified configuration.")
    # Load
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of items to process.")
    args = parser.parse_args()
    # Load configuration from YAML file
    with open(args.config_path, 'r') as file:
        config_data = yaml.safe_load(file)

    task_config = TaskConfiguration(**config_data)

    run_task(task_config, args.limit)
