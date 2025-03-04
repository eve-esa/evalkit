from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class HFModel:
    def __init__(self, model_id, tokenizer_id, batch_size=1):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, batch_size=batch_size,
                             device_map='auto')

    def generate(self, prompt: list[str], generation_args):
        # Generate predictions
        predictions = []
        for outputs in tqdm(self.pipe(prompt, return_full_text=False , **generation_args), desc="Generating"):
            # Extract generated texts from outputs
            predictions.extend([out[0]['generated_text'] for out in outputs])
        return predictions


class VllmModel:
    def __init__(self, model_id, tokenizer_id, quantization=None, load_format='auto', enforce_eager=True,
                 dtype='float16', tensor_parallel_size=1, model_kwargs: dict = None):
        if model_kwargs is None:
            model_kwargs = {}
        self.model = LLM(
            model=model_id,
            tokenizer=tokenizer_id,
            trust_remote_code=True,
            dtype=dtype,
            quantization=quantization,
            load_format=load_format,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            **model_kwargs
        )
        self.tokenizer = self.model.get_tokenizer()

    def get_sampling_params(self, generation_args) -> SamplingParams:
        sampling_params = generation_args.copy()
        if generation_args.get('top_k', None) is not None:
            sampling_params['best_of'] = generation_args['num_beams']
            del sampling_params['num_beams']
        sampling_params = SamplingParams(**sampling_params)
        return sampling_params

    def generate(self, prompts: list[str], generation_args: dict) -> list[str]:
        responses = []
        #prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        sampling_params = self.get_sampling_params(generation_args)
        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        for output in outputs:
            prompt = output.prompt
            responses.append(output.outputs[0].text)
        return responses
