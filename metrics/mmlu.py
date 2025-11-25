import os
import click


def mmlu_eval(model_path: str, output_path=None, file_name="mmlu.json"):
    if output_path is None:
        output_path = os.path.join(model_path, "evaluation")

    os.system(
        f"lm_eval --model hf --model_args pretrained={model_path} --tasks mmlu --device cuda:0 --batch_size 8 --output_path {output_path}"
    )

    model_name = model_path.split("/")[-1]
    # Workaround to get the output file since lm_eval ignores the output_path argument
    tmp_dir = [x for x in os.listdir(output_path) if model_name in x][0]
    result_file = [x for x in os.listdir(os.path.join(output_path, tmp_dir)) if "results" in x][0]

    os.system(
        f"mv {os.path.join(output_path, tmp_dir, result_file)} {os.path.join(output_path, file_name)}"
    )
    os.system(f"rm -r {os.path.join(output_path, tmp_dir)}")


@click.command()
@click.option("--model_path", help="Path to the model checkpoint")
@click.option("--output_path", default="perplexity.json", help="Path to save the perplexity value")
def main(model_path, output_path="perplexity.json"):
    mmlu_eval(model_path, output_path)


if __name__ == "__main__":
    main()
