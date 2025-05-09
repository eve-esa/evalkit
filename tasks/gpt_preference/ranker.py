import json
from random import random
from typing import Dict, Tuple

from openai import OpenAI
import os


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
            f"Output {i+1}": item["id"] for i, item in enumerate(output_list)
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
            generation = self.client.chat.completions.create(model='o4-mini', messages=messages).choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error calling ChatGPT API: {e}")
            return ""

    def parse_gpt_response(self, response: str) -> Tuple[str, Dict[str, int]]:
        """
        Parse the response from GPT to extract justification and ranking.

        Args:
            response: Raw response from ChatGPT

        Returns:
            Tuple of (justification text, ranking dictionary)
        """
        # Find the JSON portion of the response
        try:
            # First try to find JSON between triple backticks
            json_pattern = response.split("```json")
            if len(json_pattern) > 1:
                json_str = json_pattern[1].split("```")[0].strip()
            else:
                # If no triple backticks, find any JSON-like structure
                # This is a simplified approach, might need improvement
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                else:
                    raise ValueError("No JSON found in response")

            # Extract the ranking dictionary
            data = json.loads(json_str)
            ranking = data.get("ranking", {})

            # Extract justification (everything before the JSON)
            justification = response[:response.find("{")].strip()

            return justification, ranking
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

            else: # split on 'output'
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

