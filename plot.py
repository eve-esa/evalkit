import os
import json
import plotly as plt
import click

def read_scores(model_path, metric='bertscore'):
    checkpoints = get_checkpoints(model_path)
    scores = []
    for checkpoint in checkpoints:
        with open(f"{checkpoint}/evaluation/{metric}.json") as f:
            scores.append(json.load(f))
    return scores


def get_checkpoints(model_path: str):
    checkpoints_number = [int(f.name.split('-')[-1]) for f in os.scandir(model_path) if
                          f.is_dir() and 'checkpoint' in f.name]
    # sort checkpoints
    checkpoints_number.sort()
    checkpoints = [f"{model_path}/checkpoint-{checkpoint}" for checkpoint in checkpoints_number]
    return checkpoints


def plot_scores(model_path, metric='bertscore'):
    scores = read_scores(model_path, metric)
    checkpoints = get_checkpoints(model_path)
    # Plot scores on Y axis and checkpoints on X axis
    fig = plt.graph_objs.Figure()
    fig.add_trace(plt.graph_objs.Scatter(x=checkpoints, y=scores))
    # Add title and labels
    fig.update_layout(title='BertScores evolution over training',
                      xaxis_title='Checkpoints',
                      yaxis_title=metric)
    fig.show()


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--metric', help='Metric to evaluate', default='bertscore')
def main(model_path, metric='bertscore'):
    plot_scores(model_path, metric)

if __name__ == '__main__':
    plot_scores()
