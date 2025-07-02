import os
import random
import time
import yaml
from datasets import Dataset
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import Union
from tqdm import tqdm
import multiprocessing as mp
import json
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

@dataclass
class PromptData:
    prompt: str
    system_prompt: Union[str, None] = None
    examples: list = None
    dynamic_options: dict = None


def sample_dynamic_prompt(prompt_variables, dynamic_options):
    if dynamic_options is not None:
        for key, options in dynamic_options.items():
            prompt_variables[key] = random.choice(options)
    return prompt_variables


def add_few_shot_examples(prompt_variables, examples, n_shots):
    if n_shots and examples:
        selected_examples = random.sample(examples, min(n_shots, len(examples)))
        few_shot_prompt = '\n'.join(example['text'] for example in selected_examples)
        prompt_variables['examples'] = few_shot_prompt
    return prompt_variables


class ChatGroq:
    pass


def save_results(results_file, save_dict, results_lock):
    """Save results to a JSON file."""
    with results_lock:
        with open(results_file, "a+") as f:
            for save_sample in save_dict:
                json.dump(save_sample, f)
                f.write("\n")
    return results_file


def generate(sample, model_name, parser, temperature, prompt_data, parse_response,
             process_save_dict, generation_key, n_shots, max_retries, backend, vllm_url):
    """Generate text using the chosen LLM backend with DB logging and result uploading."""


    prompt_template = PromptTemplate.from_template(prompt_data.prompt)
    prompt_variables = {var: sample.get(var, '') for var in prompt_template.input_variables}

    if parser:
        prompt_variables['format_instructions'] = parser.get_format_instructions()

    prompt_variables = add_few_shot_examples(prompt_variables, prompt_data.examples, n_shots)
    prompt_variables = sample_dynamic_prompt(prompt_variables, prompt_data.dynamic_options)
    prompt = prompt_template.format_prompt(**prompt_variables)


    messages = [{"role": "user", "content": prompt}]
    if prompt_data.system_prompt:
        messages.insert(0, {"role": "system", "content": prompt_data.system_prompt})

    # Start logging to database with initial information
    prompt_str = str(prompt.text) if hasattr(prompt, 'text') else str(prompt)

    logger.debug(f"Starting generation for sample ID: {sample}")
    logger.debug(f"Using backend: {backend}")

    generation = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1} of {max_retries}")

            if backend == "groq":
                llm = ChatGroq(temperature=temperature, model=model_name)
                logger.info("Using Groq backend")
                generation = llm.invoke(prompt).content
                prompt_for_log = prompt.text if hasattr(prompt, 'text') else str(prompt)

            elif backend == "vllm":
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key="EMPTY",
                    openai_api_base=vllm_url,
                    temperature=temperature,
                )
                logger.info("Using vLLM backend")

                messages_w_role = []
                for message in messages:
                    if message["role"] == "user":
                        messages_w_role.append(HumanMessage(content=message["content"].text))
                    elif message["role"] == "system":
                        messages_w_role.append(SystemMessage(content=message["content"]))
                    else:
                        error_msg = f"Invalid role: {message['role']}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                # Format prompt for logging
                prompt_for_log = "\n".join([msg.content for msg in messages_w_role])
                generation = llm.invoke(messages_w_role).content

            elif backend == "openai":
                # Check if OpenAI API key is provided
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    error_msg = "OpenAI API key not provided"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                client = OpenAI(api_key=openai_api_key)
                logger.info("Using OpenAI backend")

                messages_w_role = []
                for message in messages:
                    new_message = {"role": None, "content": None}
                    if message["role"] == "user":
                        new_message["content"] = message["content"].text
                        new_message["role"] = "user"
                    elif message["role"] == "system":
                        new_message["content"] = message["content"]
                        new_message["role"] = "system"
                    else:
                        error_msg = f"Invalid role: {message['role']}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    messages_w_role.append(new_message)
                prompt_for_log = "\n".join([msg['content'] for msg in messages_w_role])
                generation = client.chat.completions.create(model=model_name, messages=messages_w_role).choices[0].message.content

            else:
                error_msg = f"Invalid backend: {backend}. Choose 'groq', 'vllm', or 'openai'"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not generation:
                warning_msg = "Received empty response from LLM."
                logger.warning(warning_msg)
                continue

            logger.debug("Successfully received generation")
            break

        except Exception as e:
            error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            logger.error(error_msg)

            if attempt < max_retries - 1:
                logger.info("Waiting 30 seconds before retry")
                time.sleep(30)
            else:
                error_msg = "Max retries reached. Skipping this batch."
                logger.error(error_msg)
                return None

    text_generation = generation
    try:
        logger.info("Parsing response")
        generation = parse_response(generation, logger)

        logger.info("Creating save dictionary")
        save_dict = [{generation_key: generation}]
        save_dict = process_save_dict(save_dict, generation_key, prompt_variables)

        logger_str = f"Prompt: \n{prompt_for_log} \n\nGeneration: \n{generation}"
        logger.debug(logger_str)

        logger.debug('Text generated:\n' + text_generation)

    except Exception as e:
        error_msg = f"Error in processing generation: {str(e)}"
        logger.error(error_msg)
        logger.info('Text generated:\n' + text_generation)

    return save_dict



class LLMApi:
    def __init__(self, model_name="llama3-8b-8192", output_structure=None, n_shots=3, temperature=0.1, max_retries=4,
                 prompt_path=None, results_file='../synthetic_data/qa_grades.json', parse_response=None,
                 process_save_dict=None, generation_key='generation', log_folder='logs', backend="groq",
                 vllm_url="http://localhost:8000/v1", num_workers=2, db_schema=None, db_table_name=None):
        self.n_shots = n_shots
        self.model_name = model_name
        self.max_retries = max_retries
        self.parser = PydanticOutputParser(pydantic_object=output_structure)
        self.results_file = results_file
        self.generation_key = generation_key
        self.temperature = temperature
        self.backend = backend
        self.vllm_url = vllm_url
        self.num_workers = num_workers
        self.db_schema = db_schema
        self.db_table_name = db_table_name
        #self.openai_api_key = os.getenv("OPENAI_API_KEY")

        def identity(x):
            return x

        def identity2(x, y, z):
            return x


        if parse_response is None:
            self.parse_response = identity
        else:
            self.parse_response = parse_response

        if process_save_dict is None:
            self.process_save_dict = identity2
        else:
            self.process_save_dict = process_save_dict

        # If log folder has an extension remove the .log
        if log_folder.endswith('.log'):
            log_folder = log_folder[:-4]

        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        with open(prompt_path, "r") as f:
            self.prompts = yaml.safe_load(f)
        self.prompt_data = PromptData(**self.prompts)


    def __call__(self, sample):
        return generate(sample, self.model_name, self.parser, self.temperature, self.prompt_data,
                        self.parse_response, self.process_save_dict, self.generation_key,
                        self.n_shots, self.max_retries, self.backend, self.vllm_url)


    def mp_generate(self, dataset: Dataset) -> Dataset:
        """Generate text using multiprocessing."""
        with mp.Manager() as manager:
            results_lock = manager.Lock()
            results = []

            # Create a list to collect results as they complete
            with mp.Pool(self.num_workers) as pool:
                # Create a single tqdm progress bar
                with tqdm(total=len(dataset), desc="Generating...") as pbar:
                    # Define a callback function to update progress
                    def update_progress(result):
                        pbar.update(1)
                        results.append(result)
                        save_results(self.results_file, result, results_lock)

                    # Apply async with a callback
                    jobs = [
                        pool.apply_async(
                            generate,
                            args=(sample,  self.model_name, self.parser,
                                  self.temperature, self.prompt_data, self.parse_response,
                                  self.process_save_dict, self.generation_key, self.results_file,
                                  self.n_shots, self.max_retries, self.backend,
                                  self.vllm_url),
                            callback=update_progress
                        )
                        for sample in dataset
                    ]

                    # Wait for all jobs to complete
                    for job in jobs:
                        job.wait()

        # Add the collected results as a column to the dataset
        dataset = dataset.add_column(self.generation_key, results)
        return dataset