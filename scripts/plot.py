import os
import json
import plotly as plt
import click


def load_sub_metric(data, sub_metric):
    return data[sub_metric]


def read_scores(model_path, metric='qa50', sub_metric='f1_mean'):
    checkpoints = get_checkpoints(model_path)
    scores = []
    for checkpoint in checkpoints:
        print(f"Reading scores from {checkpoint}/evaluation/{metric}.json")
        with open(f"{checkpoint}/evaluation/{metric}.json") as f:
            data = json.load(f)
            # For MMLU, we are interested in the 'groups' key
            if 'groups' in data:
                data = data['groups']
            data = load_sub_metric(data, sub_metric)
            scores.append(data)
    return scores


def get_checkpoints(model_path: str):
    checkpoints_number = [int(f.name.split('-')[-1]) for f in os.scandir(model_path) if
                          f.is_dir() and 'checkpoint' in f.name]
    # sort checkpoints
    checkpoints_number.sort()
    checkpoints = [f"{model_path}/checkpoint-{checkpoint}" for checkpoint in checkpoints_number]
    return checkpoints


def plot_scores(model_path, metric='qa50', sub_metric='f1_mean'):
    scores = read_scores(model_path, metric, sub_metric)
    checkpoints = get_checkpoints(model_path)
    checkpoints_number = [int(f.split('-')[-1]) for f in checkpoints]
    # Plot scores on Y axis and checkpoints on X axis
    fig = plt.graph_objs.Figure()
    fig.add_trace(plt.graph_objs.Scatter(x=checkpoints_number, y=scores))
    # Add title and labels
    fig.update_layout(title=f'{metric} {sub_metric} scores',
                      xaxis_title='Checkpoints',
                      yaxis_title=sub_metric,
                      width=1200,
                      height=800)
    # Save the plot
    output = f"{model_path}/plots/{metric}_{sub_metric}.png"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.write_image(output)


@click.command()
@click.option('--model_path', help='Path to the model checkpoint')
@click.option('--metric', help='Metric to evaluate', default='qa50')
@click.option('--sub_metric', help='Sub metric to evaluate', default='f1_mean')
def main(model_path, metric='qa50', sub_metric='f1_mean'):
    plot_scores(model_path, metric, sub_metric)


if __name__ == '__main__':
    main()
