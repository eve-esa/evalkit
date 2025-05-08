# lm-evaluation-harness
This folder contains the tasks compatible with lm-evaluation-harness. The tasks are organized in subfolders, each containing a task.py file and a README.md file. The README.md file contains the task description and instructions for running the task. The task.py file contains the implementation of the task.


## Clone llm-evaluation-harness
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Run  IS custom task
```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=float16,tensor_parallel_size=2,max_model_len=6000 --tasks is_open_ended --batch_size auto --output_path is_open_ended_result --log_samples --log_samples
```