import re
import logging
import json
from typing import Dict, Tuple
import random

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from pydantic import Field

from openai import OpenAI
import os

class Answer(BaseModel):
    answer: str = Field(..., description="Answer to the question.")

class GPTPreferenceRanker:
    def __init__(self):
        """
        Initialize the GPT Preference Ranker.

        Args:
            api_key: OpenAI API key for ChatGPT access
            csv_path: Path to the CSV file containing the outputs to evaluate
            task_description: Description of the task for the evaluator
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def create_evaluation_prompt(self, outputs: Dict[str, str], answer: str) -> tuple[str, dict[str, str]]:
        """
        Create an evaluation prompt with shuffled outputs.

        Args:
            outputs: Dictionary containing outputs to be evaluated

        Returns:
            Formatted prompt string with shuffled outputs
        """
        # Create a list of the outputs with their identifiers
        output_list = [
            {"id": "current_model", "content": outputs["current_model"]},
            {"id": "comparative_model", "content": outputs["comparative_model"]},
        ]

        # Shuffle the outputs
        random.shuffle(output_list)

        # Keep track of the shuffled mapping
        output_mapping = {
            f"Output {i + 1}": item["id"] for i, item in enumerate(output_list)
        }

        # Create the prompt
        prompt = f"""You are a helpful and precise evaluator for language model outputs.
        Task description (for context):
        <task_description>
        You are given some extracted parts from science papers along with a question.
        If you don't know the answer, just say "I don't know." Don't try to make up an answer.
        Use only the following pieces of context to answer the question at the end.
        Do not use any prior knowledge.
        </task_description>


        Please evaluate all outputs independently based on:
        1. Relevance to the task
        2. Accuracy
        3. Fluency and grammar
        4. Completeness of the answer

        The correct answer is the following:
        {answer}

        Return a justification text on the order of preference of all outputs based on the previous criteria.
        Then return a JSON object with:
        - "ranking": a dictionary where 1 = best, 2 = worst. Example: {{ "output_1": 1, "output_2": 2 }}

        Respond **only** with the justification and a valid JSON object.

        ### Output 1:
        {output_list[0]["content"]}

        ### Output 2:
        {output_list[1]["content"]}


        Now return output as a JSON
        """
        return prompt, output_mapping

    def call_chatgpt(self, prompt: str) -> str:
        """
        Call the ChatGPT API to evaluate the outputs.

        Args:
            prompt: Formatted prompt to send to ChatGPT

        Returns:
            ChatGPT's response
        """

        try:
            messages = [{"role": "user", "content": prompt}]
            generation = self.client.chat.completions.create(model='o4-mini', messages=messages).choices[
                0].message.content
            return generation
        except Exception as e:
            print(f"Error calling ChatGPT API: {e}")
            return ""

    def parse_gpt_response(self, response: str):
        """
        Parse the response from GPT to extract justification and ranking.

        Args:
            response: Raw response from ChatGPT

        Returns:
            Tuple of (justification text, ranking dictionary)
        """
        print(response)
        try:
            # Match any JSON object (flat or nested), capture as few chars as needed to include output_1 and output_2
            pattern = r'''
                \{
                    (?:
                        \s*"ranking"\s*:\s*       # nested format
                        (?P<nested>\{[^{}]*?\})
                        |
                        (?P<flat>[^{}]*?)         # flat format
                    )
                \}
            '''
            matches = re.finditer(pattern, response, flags=re.DOTALL | re.VERBOSE)

            for match in matches:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)

                    if "ranking" in data:
                        ranking = data["ranking"]
                    else:
                        ranking = data

                    if all(k in ranking for k in ("output_1", "output_2")):
                        justification = response[:match.start()].strip()
                        return justification, ranking
                except json.JSONDecodeError:
                    continue

            raise ValueError("No valid JSON with output_1 and output_2 found")

        except Exception as e:
            print(f"Error parsing GPT response: {e}")
            print(f"Response was: {response}")
            return response, {}


    def translate_rankings(self, ranking: Dict[str, int], mapping: Dict[str, str]) -> Dict[str, int]:
        """
        Translate the rankings from the shuffled order back to the original.

        Args:
            ranking: Ranking dictionary from GPT
            mapping: Mapping from shuffled order to original IDs

        Returns:
            Translated ranking dictionary
        """
        # Create a reverse mapping from output_x to Output N
        reverse_mapping = {}
        for output_num, output_id in mapping.items():
            # Convert "Output 1" to 1, etc.
            output_number = int(output_num.split()[1])
            reverse_mapping[output_number] = output_id

        # Translate the rankings
        translated_ranking = {}
        for output_id, rank in ranking.items():
            # Map to the original output ID (output_a, output_b, output_c)
            if '_' in output_id:
                output_number = int(output_id.split('_')[1])
                # Check if is a number or a string

            else:  # split on 'output'
                output_number = int(output_id.split('output')[1])
            output_number = int(output_id.split('_')[1])
            original_id = reverse_mapping[output_number]
            translated_ranking[original_id] = rank

        return translated_ranking

    def evaluate(self, current_model, comparative_model, correct_answer) -> int:
        """
        Return rank of the model
        """
        print("Starting GPT preference ranking evaluation...")

        # Extract the three outputs
        outputs = {
            "current_model": current_model,
            "comparative_model": comparative_model,
        }

        # Create evaluation prompt with shuffled outputs
        prompt, output_mapping = self.create_evaluation_prompt(outputs, correct_answer)

        # Call ChatGPT
        response = self.call_chatgpt(prompt)

        # Parse response
        justification, ranking = self.parse_gpt_response(response)

        # Translate rankings back to original order
        translated_ranking = self.translate_rankings(ranking, output_mapping)

        return translated_ranking['current_model']


logging.getLogger("openai").setLevel(logging.ERROR)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the file
file_path = os.path.join(script_dir, "prompt")

# Now load the file
with open(file_path, 'r') as f:
    prompt = f.read()

parser = PydanticOutputParser(pydantic_object=Answer)


def create_context_prompt(doc) -> str:
    context = doc['context']
    question = doc['question']
    formatted_prompt = prompt.format(context=context, question=question, format_instructions=parser.get_format_instructions())
    print(formatted_prompt)

    return formatted_prompt


def process_answer(answer):
    # Try JSON-style answers first
    json_answer_match = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', answer)
    if json_answer_match:
        return json_answer_match.group(1).strip()

    # Try plain format like: "Answer: ..."
    plain_answer_match = re.search(r'Answer\s*:\s*(.*)', answer)
    if plain_answer_match:
        return plain_answer_match.group(1).strip()

    # Otherwise, fallback to last non-empty line (excluding the prompt)
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return ""


def process_results(doc, results):
    model_pred = results[0]
    print('Model prediction: ', model_pred)
    model_answer = process_answer(model_pred)
    print('Model answer processed: ', model_answer)
    ranker = GPTPreferenceRanker()
    result = ranker.evaluate(model_answer, doc['answer'], doc['original_answer'])

    if result != 1:
        result = 0

    return {'win': result}


