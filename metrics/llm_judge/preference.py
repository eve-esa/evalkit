import dataclasses
import json
import re
from pathlib import Path
from typing import Optional
import random

from networkx.algorithms.core import core_number
from pydantic import BaseModel, Field
from metrics.llm_judge.judge import ModelBase
from metrics.llm_judge.llm_api import LLMApi

class LLMOutput(BaseModel):
    answer_A: int = Field(..., description="Ranking of answer A")
    answer_B: int = Field(..., description="Ranking of answer B")
    explanation: str = Field(..., description="Explanation for the ranking.")

@dataclasses.dataclass
class SampleModel:
    model_answer: str
    comparison_answer: str
    original_answer: str
    context: str = ''


class LLMPreferenceRating(ModelBase):
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
    def parse_response(response, logger) -> Optional[dict]:
        """Parse the model response to extract answer rankings and explanation."""
        response = response.strip()
        try:
            # First try to parse a valid JSON object containing all required fields
            json_match = re.search(r'\{[^{}]*"answer_A"[^{}]*"answer_B"[^{}]*"explanation"[^{}]*\}', response)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    answer_A = parsed_json.get('answer_A')
                    answer_B = parsed_json.get('answer_B')
                    explanation = parsed_json.get('explanation')

                    if isinstance(answer_A, int) and isinstance(answer_B, int) and isinstance(explanation, str):
                        return {
                            'answer_A': answer_A,
                            'answer_B': answer_B,
                            'explanation': explanation.strip()
                        }
                except json.JSONDecodeError:
                    pass

            # Fallback regex parsing
            answer_A_match = re.search(r'"?answer_A"?\s*:\s*(\d+)', response, re.IGNORECASE)
            answer_B_match = re.search(r'"?answer_B"?\s*:\s*(\d+)', response, re.IGNORECASE)
            explanation_match = re.search(r'"?explanation"?\s*:\s*"([^"]*(?:\\.[^"]*)*)"', response, re.IGNORECASE)

            if answer_A_match and answer_B_match and explanation_match:
                answer_A = int(answer_A_match.group(1))
                answer_B = int(answer_B_match.group(1))
                explanation = explanation_match.group(1)
                explanation = explanation.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')

                return {
                    'answer_A': answer_A,
                    'answer_B': answer_B,
                    'explanation': explanation.strip()
                }

            logger.debug("No valid answer_A/answer_B/explanation triplet found.")
            return None

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing response: {response}, Error: {e}")
            return None

        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing response: {response}, Error: {e}")
            return None


    def judge(self, sample: SampleModel):
        """Evaluate QA pairs using the LLM."""


        pass
        #
        # # Random sample the position of the LLM answer
        #
        # answers_list = ['answer_A', 'answer_B']
        # # Random sample
        # random.shuffle(answers_list)
        #
        # position = random.sample(
        # sample_dict = {'original_answer': sample.original_answer, 'context': sample.context}
        # if position == 0:
        #     model_answer = 'answer_A'
        #     comparison_answer = 'answer_B'
        #     sample_dict['answer_A'] = sample.model_answer
        #     sample_dict['answer_B'] = sample.comparison_answer
        #     model_answer = 'answer_A'
        #     comparison_answer = 'answer_B'
        # else:
        #     sample_dict['answer_A'] = sample.comparison_answer
        #     sample_dict['answer_B'] = sample.model_answer
        #     model_answer = 'answer_A'
        #     comparison_answer = 'answer_B'
        #
        #
        #
        # output_dict = self.llm(sample)
        # print(output_dict)
        #
        # correct = output_dict[0][self.generation_key]['correct']
        # explanation = output_dict[0][self.generation_key]['explanation']
        # return correct

    @staticmethod
    def process_save_dict(save_dict, generation_key, prompt_variables) -> list[dict]:
        """Process the save_dict and save the results."""
        save_dict = save_dict[0]

        # Extract the new fields from the generation key
        save_dict['answer_A'] = save_dict[generation_key].get('answer_A')
        save_dict['answer_B'] = save_dict[generation_key].get('answer_B')
        save_dict['explanation'] = save_dict[generation_key].get('explanation')

        return [save_dict]

