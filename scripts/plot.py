import os
import json
import plotly as plt
import click


def load_metric(data, metric):
    try:
        return data[metric]
    except KeyError:
        return data[metric+',none']


def get_results_file(eval_path, task_name):
    """
    Since different tasks could be saved together we could have a file name like 'mmlu,qa50.json'.
    This function returns the file name that contains the results for the model.
    :param output_path:
    :param model_name:
    :return:
    """
    # List all files in the output path
    files = os.listdir(eval_path)
    # Filter the files that contain the task name
    files = [f for f in files if task_name in f]
    if files:
        return files[0]
    else:
        return None


def read_metrics(model_path, task='mmlu', metric='f1_mean'):
    checkpoints = get_checkpoints(model_path)
    scores = []
    for checkpoint in checkpoints:
        eval_path = os.path.join(checkpoint, 'evaluation')
        result_file = get_results_file(eval_path, task)
        if result_file is None:
            print(f'No result file found for {checkpoint}')
            continue
        file_path = os.path.join(eval_path, result_file)
        # Read the evaluation file
        with open(file_path) as f:
            data = json.load(f)
            results = data['results'][task]
            # For MMLU, we are interested in the 'groups' key
            if 'groups' in results:
                results = results['groups']
            results = load_metric(results, metric)
            scores.append(results)
    return scores


def get_checkpoints(model_path: str):
    checkpoints_number = [int(f.name.split('-')[-1]) for f in os.scandir(model_path) if
                          f.is_dir() and 'checkpoint' in f.name]
    # sort checkpoints
    checkpoints_number.sort()
    checkpoints = [f"{model_path}/checkpoint-{checkpoint}" for checkpoint in checkpoints_number]
    return checkpoints


import plotly.graph_objects as go


def plot_comparison(models, task='mmlu,is_mcqa', metric='acc', output_path=None, names=None, title=None):
    """
    Plot the comparison between the models using Plotly.
    :param models: List of model names
    :param task: The task name
    :param metric: The evaluation metric
    :param output_path: If specified, saves the figure
    """
    fig = go.Figure()  # Initialize a figure
    if names is None:
        names = models[0].split('/')[-1]
    if title is None:
        title = f'{task.capitalize()} {metric} scores'

    for model, name in zip(models, names):
        scores = read_metrics(model, task, metric)
        checkpoints = get_checkpoints(model)
        checkpoints_number = [int(f.split('-')[-1]) for f in checkpoints]

        # Add trace for each model
        fig.add_trace(go.Scatter(x=checkpoints_number, y=scores, mode='lines+markers', name=name))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Checkpoints',
        yaxis_title=metric,
        width=1200,
        height=800
    )

    if output_path is None:
        # Save the plot
        output_path = f"./plots/{task}_{metric}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path)  # Save as an image if output_path is provided




def plot_scores(model_path, task='mmlu', metric='acc', output_path=None, name=None, title=None):
    if name is None:
        name = model_path.split('/')[-1]
    if title is None:
        title = f'Model: {name}\n{task.capitalize()} {metric} scores'
    scores = read_metrics(model_path, task, metric)
    checkpoints = get_checkpoints(model_path)
    checkpoints_number = [int(f.split('-')[-1]) for f in checkpoints]
    # Plot scores on Y axis and checkpoints on X axis
    fig = plt.graph_objs.Figure()
    fig.add_trace(plt.graph_objs.Scatter(x=checkpoints_number, y=scores))
    # Add title and labels
    fig.update_layout(title=title,
                      xaxis_title='Checkpoints',
                      yaxis_title=metric,
                      width=1200,
                      height=800)

    if output_path is  None:
        # Save the plot
        output_path = f"{model_path}/plots/{task}_{metric}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path)


@click.command()
@click.option('--model_path', help='Path to the model checkpoint. If more separated by comma')
@click.option('--task', help='Task to plot', default='mmlu')
@click.option('--metric', help='Metric to plot', default='f1_mean')
@click.option('--output_path', help='Output path')
@click.option('--names', help='Names of the model in the comparison')
@click.option('--title', help='Title of the plot', default='Amazing plot')
def main(model_path, task='mmlu', metric='acc', output_path=None, names=None, title=None):
    if ',' in model_path:
        models = model_path.split(',')
        if names is not None:
            names = names.split(',')
        plot_comparison(models, task, metric, output_path, names, title)
    else:
        plot_scores(model_path, task, metric, output_path, names, title)


if __name__ == '__main__':
    main()
