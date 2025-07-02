import json
import re
from pathlib import Path

from networkx.algorithms.core import core_number
from pydantic import BaseModel, Field
from metrics.llm_judge.judge import ModelBase
from metrics.llm_judge.llm_api import LLMApi

class LLMOutput(BaseModel):
    correct: bool = Field(..., description="True if the answer is correct, False otherwise.")
    critique: str = Field(..., description="Critiques for the QA pair.")


class LLMCorrectnessEvaluator(ModelBase):
    """Evaluator using an LLM as a judge."""

    def __init__(self, model_name="gpt-4o-mini-2024-07-18", temperature=0.1, max_retries=3, prompt_path=None, results_file=None, n_shots=2, backend="openai", num_workers=1):
        super().__init__(results_file)
        if prompt_path is None:
            prompt_path = Path(__file__).parent / 'prompts/grade_qa.yaml'

        self.prompt_path = prompt_path
        self.num_workers = num_workers
        self.generation_key = 'answer'

        self.output_structure = LLMOutput
        self.llm = LLMApi(model_name=model_name, temperature=temperature, parse_response=self.parse_response, process_save_dict=self.process_save_dict, prompt_path=prompt_path, output_structure=self.output_structure, results_file=self.results_file, n_shots=n_shots, generation_key=self.generation_key, backend=backend, num_workers=num_workers)
        self.results_file = results_file

    @staticmethod
    def parse_response(response, logger):
        """Parse the model response to extract correct boolean and critique."""
        response = response.strip()
        try:
            # First try to parse as valid JSON
            json_match = re.search(r'\{[^{}]*"correct"[^{}]*"critique"[^{}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    correct = parsed_json.get('correct')
                    critique = parsed_json.get('critique')

                    if correct is not None and critique is not None:
                        return {'correct': correct, 'critique': critique}
                except json.JSONDecodeError:
                    pass

            # Fallback to regex parsing if JSON parsing fails
            correct_match = re.search(r'"correct":\s*(true|false)', response, re.IGNORECASE)
            critique_match = re.search(r'"critique":\s*"([^"]*(?:\\.[^"]*)*)"', response)

            if correct_match and critique_match:
                correct = correct_match.group(1).lower() == 'true'
                critique = critique_match.group(1)
                # Handle escaped characters in critique
                critique = critique.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
                return {'correct': correct, 'critique': critique}
            else:
                logger.debug("No valid correct/critique pair found.")
                return None

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing response: {response}, Error: {e}")
            return None


    def judge(self, sample: dict):
        """Evaluate QA pairs using the LLM."""
        output_dict = self.llm(sample)

        correct = output_dict[0][self.generation_key]['correct']
        critique = output_dict[0][self.generation_key]['critique']

        if correct:
            correct = 1
        else:
            correct = 0
        return correct

    @staticmethod
    def process_save_dict(save_dict, generation_key, prompt_variables)->list[dict]:
        """Process the save_dict and save the results."""
        save_dict = save_dict[0]
        save_dict['critique'] = save_dict[generation_key]['critique']
        save_dict['correct'] = save_dict[generation_key]['correct']
        return [save_dict]
