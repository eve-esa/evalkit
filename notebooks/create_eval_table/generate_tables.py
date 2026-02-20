#!/usr/bin/env python3
"""
Script to generate evaluation tables from CSV files based on YAML configuration.
"""

import os
import argparse
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_and_merge_data(input_files, base_dir=None):
    """Load and merge CSV files."""
    dfs = []
    for file_path in input_files:
        if base_dir:
            file_path = os.path.join(base_dir, file_path)
        df = pd.read_csv(file_path)
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def apply_filters(df, config):
    """Apply filters to the dataframe based on configuration."""
    filters = config.get('filters', {})

    # Remove stderr rows
    if filters.get('remove_stderr', True):
        df = df[~df['metric'].str.contains('stderr', na=False)]

    # Remove mmlu_pro subtasks
    if filters.get('remove_mmlu_pro_subtasks', True):
        df = df[~df['subtask'].str.contains('mmlu_pro_', na=False)]

    # Filter by models
    models_config = config.get('models', {})
    if not models_config.get('all', False):
        selected_models = models_config.get('selected', [])
        if selected_models:
            df = df[df['model_name'].isin(selected_models)]

    # Filter by tasks
    tasks_config = config.get('tasks', {})
    if not tasks_config.get('all', True):
        selected_tasks = tasks_config.get('selected', [])
        if selected_tasks:
            df = df[df['task'].isin(selected_tasks)]

    return df


def create_task_dict(df):
    """Create a dictionary where the key is the task and the value is the corresponding subset of the df."""
    task_dict = {task: df[df['task'] == task] for task in df['task'].unique()}
    return task_dict


def prepare_table_df(task_df, task_name, config):
    """Prepare a dataframe for table generation."""
    # Remove 'task' and 'subtask' columns
    table_df = task_df.drop(columns=['task', 'subtask']).copy()

    # Filter metrics for hallucination detection tasks
    if 'hallucination_detection' in task_name:
        filters = config.get('filters', {})
        metrics = filters.get('hallucination_detection_metrics', ['f1', 'acc', 'precision', 'recall'])
        table_df = table_df[table_df['metric'].isin(metrics)]

    # Convert value column to percentage
    options = config.get('options', {})
    if options.get('values_as_percentage', True):
        decimal_places = options.get('decimal_places', 2)
        table_df['value'] = (table_df['value'] * 100).round(decimal_places)

    # Sort by value
    if options.get('sort_by_value', True):
        table_df = table_df.sort_values('value', ascending=False)

    return table_df


def generate_latex_tables(task_dict, output_dir, config):
    """Generate individual LaTeX tables for each task."""
    options = config.get('options', {})
    decimal_places = options.get('decimal_places', 2)

    for task_name, task_df in task_dict.items():
        table_df = prepare_table_df(task_df, task_name, config)

        # Convert to LaTeX
        latex_table = table_df.to_latex(
            index=False,
            caption=f'{task_name}',
            label=f'tab:{task_name.replace("_", "-")}',
            position='htbp',
            escape=False,
            float_format=f'%.{decimal_places}f'
        )

        # Save to file
        filename = os.path.join(output_dir, f'{task_name}.tex')
        with open(filename, 'w') as f:
            f.write(latex_table)

        print(f'Saved {filename}')


def generate_latex_recap(task_dict, output_dir, config):
    """Generate combined LaTeX recap file with all tables."""
    options = config.get('options', {})
    bold_max = options.get('bold_max_value', True)

    recap_content = []

    for task_name, task_df in task_dict.items():
        table_df = prepare_table_df(task_df, task_name, config)

        # Bold maximum value if enabled
        if bold_max:
            max_value = table_df['value'].max()
            decimal_places = options.get('decimal_places', 2)
            table_df['value'] = table_df['value'].apply(
                lambda x: f'\\textbf{{{x:.{decimal_places}f}}}' if x == max_value else f'{x:.{decimal_places}f}'
            )

        # Convert to LaTeX
        latex_table = table_df.to_latex(
            index=False,
            caption=f'{task_name}',
            label=f'tab:recap-{task_name.replace("_", "-")}',
            position='htbp',
            escape=False
        )

        recap_content.append(latex_table)

    # Combine all tables
    full_recap = '\n\n'.join(recap_content)

    # Save to recap.tex
    filename = os.path.join(output_dir, 'recap.tex')
    with open(filename, 'w') as f:
        f.write(full_recap)

    print(f'Saved {filename} with all tables combined')


def generate_markdown_report(df, task_dict, output_dir, config):
    """Generate markdown report."""
    options = config.get('options', {})
    bold_max = options.get('bold_max_value', True)
    decimal_places = options.get('decimal_places', 2)

    md_content = []

    # Add title and header
    md_content.append("# Model Evaluation Results Report\n")
    md_content.append(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

    # Add summary section
    md_content.append("## Summary\n")
    md_content.append(f"This report contains evaluation results for {len(df['model_name'].unique())} models across {len(task_dict)} different tasks.\n\n")
    md_content.append("### Models Evaluated:\n")
    for model in sorted(df['model_name'].unique()):
        md_content.append(f"- {model}\n")
    md_content.append("\n")

    # Add tasks section
    md_content.append("### Tasks:\n")
    for task in sorted(task_dict.keys()):
        md_content.append(f"- {task}\n")
    md_content.append("\n---\n\n")

    # Add detailed results for each task
    for task_name in sorted(task_dict.keys()):
        table_df = prepare_table_df(task_dict[task_name], task_name, config)

        # Find the best model
        if not table_df.empty:
            max_value = table_df['value'].max()
            best_idx = table_df['value'].idxmax()
            best_model = table_df.loc[best_idx, 'model_name']
            best_value = table_df.loc[best_idx, 'value']

            # Format the value column with bold for the maximum values
            if bold_max:
                table_df['value'] = table_df['value'].apply(
                    lambda x: f'**{x:.{decimal_places}f}**' if x == max_value else f'{x:.{decimal_places}f}'
                )

            # Add task section
            md_content.append(f"## {task_name.replace('_', ' ').title()}\n\n")
            md_content.append(f"**Best Model:** {best_model} ({best_value:.{decimal_places}f}%)\n\n")

            # Convert to markdown table
            md_table = table_df.to_markdown(index=False)
            md_content.append(md_table)
            md_content.append("\n\n")

    # Combine all content
    full_md = ''.join(md_content)

    # Determine filename based on model filtering
    models_config = config.get('models', {})
    if models_config.get('all', False):
        filename = 'evaluation_report.md'
    else:
        filename = 'evaluation_report_filtered.md'

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        f.write(full_md)

    print(f'Saved {filepath}')


def generate_excel_file(df, task_dict, output_dir, config):
    """Generate Excel file with tabs for each task."""
    # Determine filename based on model filtering
    models_config = config.get('models', {})
    if models_config.get('all', False):
        filename = 'evaluation_results.xlsx'
    else:
        filename = 'evaluation_results_filtered.xlsx'

    filepath = os.path.join(output_dir, filename)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Create a summary sheet
        summary_data = {
            'Total Models': [len(df['model_name'].unique())],
            'Total Tasks': [len(task_dict)],
            'Generated On': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Add a sheet listing all models
        models_df = pd.DataFrame({'Model Name': sorted(df['model_name'].unique())})
        models_df.to_excel(writer, sheet_name='Models', index=False)

        # Create a sheet for each task
        for task_name in sorted(task_dict.keys()):
            table_df = prepare_table_df(task_dict[task_name], task_name, config)

            # Create a safe sheet name (Excel has 31 char limit)
            sheet_name = task_name[:31].replace('/', '_').replace('\\', '_').replace('*', '_').replace('[', '_').replace(']', '_').replace(':', '_').replace('?', '_')

            # Write to Excel
            table_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f'Saved {filepath}')
    print(f'Excel file contains {len(task_dict)} task sheets plus Summary and Models sheets')


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation tables from CSV files')
    parser.add_argument('--config', '-c', default='config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--base-dir', '-d', default=None, help='Base directory for input files (default: config file directory)')

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    config = load_config(config_path)
    print(f"Loaded configuration from {args.config}")

    # Determine base directory
    base_dir = args.base_dir or config_path.parent

    # Load and merge data
    print("\nLoading data files...")
    input_files = config.get('input_files', [])
    df = load_and_merge_data(input_files, base_dir)
    print(f"Loaded {len(df)} rows from {len(input_files)} files")

    # Apply filters
    print("\nApplying filters...")
    df = apply_filters(df, config)
    print(f"After filtering: {len(df)} rows, {len(df['model_name'].unique())} models, {len(df['task'].unique())} tasks")

    # Create task dictionary
    task_dict = create_task_dict(df)

    # Create output directory
    output_dir = config.get('output_dir', 'tables')
    output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate outputs based on configuration
    outputs = config.get('outputs', {})

    print("\nGenerating outputs...")

    if outputs.get('latex_individual', False):
        print("\n--- Generating individual LaTeX tables ---")
        generate_latex_tables(task_dict, output_dir, config)

    if outputs.get('latex_recap', False):
        print("\n--- Generating LaTeX recap ---")
        generate_latex_recap(task_dict, output_dir, config)

    if outputs.get('markdown', False):
        print("\n--- Generating markdown report ---")
        generate_markdown_report(df, task_dict, output_dir, config)

    if outputs.get('excel', False):
        print("\n--- Generating Excel file ---")
        generate_excel_file(df, task_dict, output_dir, config)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()