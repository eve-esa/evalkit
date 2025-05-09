# eve-evaluation
Use the correct docker template for your environment


Clone the repo
```bash
git clone -b dev https://github.com/eve-esa/eve-evaluation.git
````

Copy models weights from s3
```bash
aws s3 cp s3://llm4eo-s3/eve_checkpoint_data/{your_folder} . --recursive --exclude "*.pt"

```

# Init environment
Init local env
```bash
python -m venv venv
source venv
```

Install requirements
```bash
pip install -r requirements.txt
```
Install llm-evaluation-harness
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install -e .[vllm]
```
To run lm_eval as it is
```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=float16,tensor_parallel_size=2,max_model_len=6000 --tasks is_open_ended --batch_size auto --output_path is_open_ended_result --log_samples --log_samples
```