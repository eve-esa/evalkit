from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import os

@dataclass
class EvaluationResult:
    score: float
    critique: Union[str, None]


class ModelBase(ABC):
    def __init__(self, results_file=None):
        self.results_file = results_file

        if results_file is not None:
            # Create all dirs in results_file path
            if os.path.dirname(self.results_file):
                os.makedirs(os.path.dirname(self.results_file), exist_ok=True)


    @abstractmethod
    def judge(self, sample: dict):
        pass

    @staticmethod
    def process_save_dict(save_dict, generation_key, prompt_variables):
        pass

    @staticmethod
    def parse_response(response, logger):
        pass