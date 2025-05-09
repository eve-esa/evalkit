# eve-evaluation
data, notebooks, and scripts to evaluate eve at various training stages

## Clone the dev branch
```bash
git clone -b dev https://github.com/eve-esa/eve-evaluation.git
````

## Copy models weights from s3
```bash
aws s3 cp s3://llm4eo-s3/eve_checkpoint_data/{your_folder} . --recursive --exclude "*.pt"

```

