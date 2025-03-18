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
    # Return the first file
    return files[0]




def read_metrics(model_path, task='mmlu', metric='f1_mean'):
    checkpoints = get_checkpoints(model_path)
    scores = []
    for checkpoint in checkpoints:
        file_path = os.path.join(model_path, checkpoint, 'evaluation', get_results_file(model_path, task))
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



def plot_comparison(models, task='mmlu,is_mcqa', metric='acc', output_path=None):
    """
    Plot the comparison between the models
    :param models:
    :param metric:
    :param sub_metric:
    :return:
    """
    for model in models:
        scores = read_metrics(model, task, metric)
        checkpoints = get_checkpoints(model)
        checkpoints_number = [int(f.split('-')[-1]) for f in checkpoints]
        # Plot scores on Y axis and checkpoints on X axis
        plt.graph_objs.Figure.add_trace(plt.graph_objs.Scatter(x=checkpoints_number, y=scores, name=model))
    # Add title and labels
    plt.graph_objs.Figure.update_layout(title=f'{task} {metric} scores',
                                        xaxis_title='Checkpoints',
                                        yaxis_title=metric,
                                        width=1200,
                                        height=800)

    if output_path is None:
        # Save the plot
        output_path = f"./plots/{task}_{metric}.png"
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.graph_objs.Figure.write_image(output_path)



def plot_scores(model_path, task='mmlu', metric='acc', output_path=None):
    scores = read_metrics(model_path, task, metric)
    checkpoints = get_checkpoints(model_path)
    checkpoints_number = [int(f.split('-')[-1]) for f in checkpoints]
    # Plot scores on Y axis and checkpoints on X axis
    fig = plt.graph_objs.Figure()
    fig.add_trace(plt.graph_objs.Scatter(x=checkpoints_number, y=scores))
    # Add title and labels
    fig.update_layout(title=f'{task} {metric} scores',
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
def main(model_path, task='mmlu', metric='acc', output_path='plots'):
    if ',' in model_path:
        models = model_path.split(',')
        plot_comparison(models, task, metric, output_path)
    plot_scores(model_path, task, metric, output_path)


if __name__ == '__main__':
    main()
