import json
from typing import Dict, Type, Any, List, Optional

from litellm import OpenAI, batch_completion
from pydantic import BaseModel, create_model

from loguru import logger

logger.debug("That's it, beautiful and simple logging!")


_SCALARS: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}

def _convert_schema_to_model(name: str, spec: Dict[str, Any]) -> Type[BaseModel]:
    fields: Dict[str, Any] = {}

    def _assert_type_is_supported(type_name: str) -> None:
        if type_name not in _SCALARS:
            raise TypeError(f"Unsupported scalar type '{type_name}'. Allowed: {', '.join(_SCALARS)}")

    for field_name, node in spec.items():
        if isinstance(node, str):
            _assert_type_is_supported(node)
            fields[field_name] = _SCALARS[node]

        elif isinstance(node, dict):
            fields[field_name] = _convert_schema_to_model(f"{name}_{field_name}", node)

        elif isinstance(node, list):
            if len(node) != 1:
                raise ValueError(f"List spec for '{field_name}' must have exactly one element.")
            item_spec = node[0]
            if isinstance(item_spec, dict):
                item_type = _convert_schema_to_model(f"{name}_{field_name}_item", item_spec)
            else:
                _assert_type_is_supported(item_spec)
                item_type = _SCALARS[item_spec]
            fields[field_name] = List[item_type]
        else:
            raise TypeError(f"Field '{field_name}' has unsupported spec node type: {type(node).__name__}")

    return create_model(name, **fields)


class ApiModel:
    def __init__(self, model_name, response_model, api_key, base_url, prompt, prompt_response_schema_key="response_schema_json", temperature=0.7, max_tokens=10_000, batch_size=10):
        self.base_url = None

        self._logger = logger

        self.response_model = response_model
        self.api_key = api_key
        self.base_url = base_url
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._response_model_cls: Optional[Type[BaseModel]] = _convert_schema_to_model("ResponseModel", response_model)
        self.model_name = model_name
        self.prompt = prompt
        self.params = {
            "model": self.model_name,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": self._response_model_cls
        }
        self.prompt_response_schema_key = prompt_response_schema_key
        self.batch_size = 10

    def _process_batch(self, inputs: List[dict[str, str]]) -> List[dict[str, str]]:
        """Processes a list of records using litellm.batch_completion."""
        messages_list = []
        for single_input in inputs:
            if self._response_model_cls:
                single_input[self.prompt_response_schema_key] = json.dumps(
                    self._response_model_cls.model_json_schema(), indent=2
                )
            formatted_prompt = self.prompt.format(**single_input)
            self._logger.debug(formatted_prompt)
            messages_list.append([{"role": "user", "content": formatted_prompt}])

        responses = batch_completion(messages=messages_list, **self.params)

        outputs = []
        for single_input, response in zip(inputs, responses):
            if isinstance(response, Exception):
                self._logger.error(f"LLM batch completion failed for an item: {response}")
                raise response

            content_str = response.choices[0].message.content or ""
            if content_str.strip():
                response_data = json.loads(content_str)
            else:
                response_data = {}
            outputs.append(response_data)
        return outputs

    def process_batch(self, inputs: List[dict[str, str]]) -> List[dict]:
        """Processes a batch of inputs and returns the results."""
        if not inputs:
            return []

        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results