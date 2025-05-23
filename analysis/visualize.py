

"""
1. Percentage of sentences flipped sentiment, kept the same, became neutral.
    One column is humans, each other columns is one of the models.
    3 plots - positive, negative, combined.

2. Agreement between models -- heatmap. Last row is human annotations.
    maybe we want 3 heatmaps -- positive, negative, combined.

3. SOTA sentiment analysis model - plot the histogram of positive scores before and after framing.
"""
import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

company_names = ['google', 'mistralai', 'deepseek-ai', 'meta-llama']
MAP_MODELS_TO_COLORS = {
        "gemma2-2b": "#FFB3B3",
        "gemma-2-9b": "#FFB3B3",  # Light Red
        "gemma-2-9b-with-neutral": "#FFB3B3",
        "gemma-2-27b": "#FF0000",  # Dark Red
        "mistral-7b-v0.3": "#ADD8E6",  # Light Blue
        "mistral-7b-v0.3-try1": "#ADD8E6",  # Light Blue
        "mistral-7b-v0.3-try2": "#ADD8E6",  # Light Blue
        "mistral-7b-v0.3-with-neutral": "#ADD8E6",  # Light Blue
        "mistral-7b-v0.3-from-ollama": "#ADD8E6",
        "mixtral-8x7b-v0.1": "#6495ED",  # Medium Blue
        "mixtral-8x22b-v0.1": "#0000FF",  # Dark Blue
        "deepseek-67b": "#CBA6FF",  # Medium Purple
        "gpt-4o": "#CBA6FF",  # Medium Purple
        "qwen2.5-7b": "#FFD580",  # Light Orange
        "qwen2.5-14b": "#FFA500",  # Medium Orange
        "qwen2.5-72b": "#FF8C00",  # Dark Orange
        "llama-3-8b": "#90EE90",  # Light Green
        "llama-3-8b-with-neutral": "#90EE90",  # Light Green
        "llama3.1-8b": "#90EE90",
        "llama-3-70b": "#008000",  # Dark Green
        "phi3.5-3.8b": "#FF69B4",  # Medium Pink
    }


def generate_labels_csv(models_dir, human_annotations_path):
    human_annotations = pd.read_csv(human_annotations_path)
    sentence_ids = []
    sentence_base_sentiment = []
    human_scores = []
    for i, row in human_annotations.iterrows():
        sentence_ids.append(row['sentence_id'])
        sentence_base_sentiment.append(row['base_sentiment'])
        if row['sentiment_flip_after_framing']:
            human_scores.append(row['majority_confidence'])
        else:
            human_scores.append(1-row['majority_confidence'])
    df_dict = {'sentence_id': sentence_ids, 'base_sentiment': sentence_base_sentiment, 'humans': human_scores}
    models_names = []
    for file_name in os.listdir(models_dir):
        name = get_model_name_for_print(file_name)
        models_names.append(name)
        model_df = pd.read_csv(os.path.join(models_dir, file_name))
        current_model_scores = []
        for sentence_id in sentence_ids:
            row = model_df[model_df['ID'] == sentence_id].iloc[0]
            opposite_framing_sentiment_pred = row['opposite_framing_processed_pred']
            original_sentiment = row['answer']
            if original_sentiment == opposite_framing_sentiment_pred:  # not flipped
                current_model_scores.append(0)
            elif opposite_framing_sentiment_pred == 'neutral':
                current_model_scores.append(0.5)
            else:  # flipped
                current_model_scores.append(1)
        df_dict[name] = current_model_scores

    df_scores = pd.DataFrame(df_dict)
    # sort columns
    df_scores = df_scores[['sentence_id', 'base_sentiment', 'humans'] + sorted(models_names)]
    return df_scores


def get_color(value):
    norm = Normalize(vmin=0, vmax=8)
    return plt.cm.viridis(norm(7-value))


def mean_absolute_distance(v1, v2):
    return np.mean(np.abs(v1 - v2)).round(2)


def run_analysis(models_dir, human_annotations_path, out_dir):
    df_scores = generate_labels_csv(models_dir, human_annotations_path)

    # capitalize column names from third colum
    df_scores.columns = df_scores.columns[:3].to_list() + [col.capitalize() for col in df_scores.columns[3:]]
    correlation_matrix = df_scores.iloc[:, 2:].corr()
    only_model_correlation = correlation_matrix[1:].drop(columns=['humans'])
    only_humans_correlation = correlation_matrix['humans'].drop(index=['humans'])


    # bar chart for humans
    colors = ["#6495ED" for _ in only_model_correlation.index]
    fig, ax = plt.subplots(figsize=(6, 4))
    only_humans_correlation = only_humans_correlation.iloc[::-1]
    bars = ax.barh(only_humans_correlation.index, only_humans_correlation.values, color=colors)
    # Add text labels with the values
    for bar in bars:
        width = bar.get_width()  # Get the width of each bar (corresponding to the value)
        ax.text(
            width - 0.07,  # Slightly offset the text from the bar's end
            bar.get_y() + bar.get_height() / 2,  # Center vertically
            f"{width:.2f}",  # Format the value to 2 decimal places
            va='center',  # Align vertically
            ha='left'  # Align horizontally
        )
    ax.set_xlabel("Pearson Correlation Coefficient with Human Behavior")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_with_humans.png'))

    # heatmap for models
    mask = np.triu(np.ones_like(only_model_correlation, dtype=bool), k=1)    # Upper triangle and diagonal mask
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(only_model_correlation, mask=mask, cmap="Reds", annot=True, fmt=".2f", vmin=0, vmax=1,
                cbar=True, square=True, linewidths=0.3, cbar_kws={'label': 'Correlation Coefficient'})

    ax.set_xticks(np.arange(len(only_model_correlation)) + 0.5)  # Set tick positions to cell centers
    labels_x = only_model_correlation.columns.to_list()
    ax.set_xticklabels(labels_x, rotation=90, ha='center')  # Set x-axis tick labels
    ax.set_yticks(np.arange(len(only_model_correlation)) + 0.5)  # Set tick positions to cell centers
    labels_y = only_model_correlation.columns.to_list()
    ax.set_yticklabels(labels_y,  rotation=0, va='center')  # Set y-axis tick labels
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairwise_correlation_matrix.png'))

    model_names = sorted(list(df_scores.columns[3:]))
    all_model_results = []
    group_by_original_sentiment = df_scores.groupby('base_sentiment')

    for model in model_names:
        df_model_positive = group_by_original_sentiment.get_group('positive')[['sentence_id', model]]
        orig_positive_flipped = df_model_positive[df_model_positive[model] == 1]['sentence_id'].to_list()
        orig_positive_became_neutral = df_model_positive[df_model_positive[model] == 0.5]['sentence_id'].to_list()
        orig_positive_kept = df_model_positive[df_model_positive[model] == 0]['sentence_id'].to_list()

        df_model_negative = group_by_original_sentiment.get_group('negative')[['sentence_id', model]]
        orig_negative_flipped = df_model_negative[df_model_negative[model] == 1]['sentence_id'].to_list()
        orig_negative_became_neutral = df_model_negative[df_model_negative[model] == 0.5]['sentence_id'].to_list()
        orig_negative_kept = df_model_negative[df_model_negative[model] == 0]['sentence_id'].to_list()

        total_flipped = orig_positive_flipped + orig_negative_flipped
        total_became_neutral = orig_positive_became_neutral + orig_negative_became_neutral
        total_kept = orig_positive_kept + orig_negative_kept

        all_model_results.append(
            {'model': model, 'orig_positive_flipped': orig_positive_flipped,
             'orig_negative_flipped': orig_negative_flipped,
             'orig_positive_became_neutral': orig_positive_became_neutral,
             'orig_negative_became_neutral': orig_negative_became_neutral,
             'orig_positive_kept': orig_positive_kept,
             'orig_negative_kept': orig_negative_kept, 'total_flipped': total_flipped,
             'total_became_neutral': total_became_neutral, 'total_kept': total_kept,
             'negative_flipped_minus_positive_flipped': len(orig_negative_flipped) - len(orig_positive_flipped),})

    plot_model_distribution(all_model_results, model_names, key='total', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_positive', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_negative', dir_path=out_dir)

    # humans distribution
    positive_humans = (group_by_original_sentiment.get_group('positive')['humans']*5).apply(round)
    flipped_positive = positive_humans.value_counts()
    negative_humans = (group_by_original_sentiment.get_group('negative')['humans']*5).apply(round)
    flipped_negative = negative_humans.value_counts()

    plot_differences(all_model_results, model_names, out_dir, humans_results=
        {'positive_flipped': np.sum((flipped_positive[5],flipped_positive[4],flipped_positive[3])),
         'negative_flipped': np.sum((flipped_negative[5],flipped_negative[4],flipped_negative[3]))})


    flipped_percentage = [[],[]]
    total_positive = len(df_scores[df_scores['base_sentiment'] == 'positive'])
    total_negative = len(df_scores[df_scores['base_sentiment'] == 'negative'])
    for i in range(6):
        flipped_percentage[0].append(round(flipped_positive[i] / total_positive * 100, 1))
        flipped_percentage[1].append(round(flipped_negative[i] / total_negative * 100, 1))
    for j in range(2):
        flipped_percentage[j].reverse()
    flipped_percentage = np.array(flipped_percentage)

    categories = [f"{i}" for i in range(6)]
    col_names = ['Positive Base', 'Negative Base']
    categories.reverse()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = []
    lefts = np.zeros(len(col_names))

    for i, category in enumerate(categories):
        color = get_color(i+1)

        bar = ax.barh(col_names, flipped_percentage[:, i], left=lefts, label=category, color=color)
        bars.append(bar)

        # Adding text on bars
        for j, value in enumerate(flipped_percentage[:, i]):
            if value == 0:
                continue
            ax.text(lefts[j] + value / 2, j, f'{value}%', ha='center', va='center', fontsize=14)

        lefts += flipped_percentage[:, i]

    # Adding labels and title
    ax.set_xlabel('% Sentences', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # set y label
    ax.legend(loc='upper right', bbox_to_anchor=(0.9, 1.20), ncol=len(categories), fontsize=14, title="Number of Sentiment Shifts", title_fontsize=14)
    plt.tight_layout()

    # Show the plot
    path = os.path.join(out_dir, f'humans_distribution.png')
    plt.savefig(path)




def get_model_name_for_print(file_name):
    name = file_name[:file_name.find('_')]
    name = name.lower()
    if 'fp16' in name:
        name = name[:name.find('-fp16')]
    if '2024' in name:
        name = name[:name.find('-2024')]
    for symbol in ['-instruct', '-it', '-mini', '-chat', '-llm', '-hf']:
        name = name.replace(symbol, '')
    name = name.replace(':', '-')
    for company in company_names:
        if company in name:
            name = name.replace(company + '-', '')
            break
    return name


def plot_model_distribution(all_model_results, model_names, key, dir_path):
    all_models_total = []
    for i, model in enumerate(model_names):
        total_flipped_for_plot = len(all_model_results[i][f'{key}_flipped'])
        total_kept_for_plot = len(all_model_results[i][f'{key}_kept'])
        total_to_neutral_for_plot = len(all_model_results[i][f'{key}_became_neutral'])
        total_num_samples = total_kept_for_plot + total_to_neutral_for_plot + total_flipped_for_plot
        # make this as a percentage
        total_kept_for_plot = round(total_kept_for_plot / total_num_samples * 100, 1)
        total_to_neutral_for_plot = round(total_to_neutral_for_plot / total_num_samples * 100, 1)
        total_flipped_for_plot = round(100 - total_kept_for_plot - total_to_neutral_for_plot, 1)
        # print(total_kept_for_plot + total_to_neutral_for_plot + total_flipped_for_plot)
        # assert total_kept_for_plot + total_to_neutral_for_plot + total_flipped_for_plot == 100
        all_models_total.append([total_flipped_for_plot, total_to_neutral_for_plot, total_kept_for_plot])
    all_models_total = np.array(all_models_total)
    categories = ['Sentiment Shift', 'Became Neutral', 'Base Sentiment Remains']
    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = []
    lefts = np.zeros(len(model_names))
    for i, category in enumerate(categories):
        norm = Normalize(vmin=0, vmax=8)
        value = (i * 2.5) + 1
        color = plt.cm.viridis(norm(7 - value))
        bar = ax.barh(model_names, all_models_total[:, i], left=lefts, label=category, color=color)
        bars.append(bar)

        # Adding text on bars
        for j, value in enumerate(all_models_total[:, i]):
            if value == 0:
                continue
            ax.text(lefts[j] + value / 2, j, f'{value}%', ha='center', va='center', fontsize=10)

        lefts += all_models_total[:, i]
    # Adding labels and title
    ax.set_xlabel('% Sentences', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(categories), fontsize=12)
    plt.tight_layout()
    # Show the plot
    path = os.path.join(dir_path, f'{key}_models_distribution.png')
    plt.savefig(path)


def plot_differences(all_model_results, model_names, dir_path, humans_results):

    # Define bar width and spacing
    bar_width = 0.8
    bar_spacing = 0.5
    x = np.arange(len(model_names) + 1) * (bar_width * 2 + bar_spacing)  # Adjust positions with spacing

    # Create figure
    fig, ax = plt.subplots(figsize=(13, 9))

    # Compute data
    all_models_positive_shifts = [len(model['orig_positive_flipped']) / 500 * 100 for model in all_model_results]
    all_models_negative_shifts = [len(model['orig_negative_flipped']) / 500 * 100 for model in all_model_results]
    humans_positive_shifts = humans_results['positive_flipped'] / 500 * 100
    humans_negative_shifts = humans_results['negative_flipped'] / 500 * 100

    # Plot vertical bars with spacing
    ax.bar(x[:-1] + bar_width, all_models_negative_shifts, width=bar_width, label="Negative", color="lightcoral")
    ax.bar(x[:-1], all_models_positive_shifts, width=bar_width, label="Positive", color="lightgreen")
    mean_models_pos_shift = np.mean(all_models_positive_shifts)
    mean_models_neg_shift = np.mean(all_models_negative_shifts)
    # plot horizontal line for mean
    ax.axhline(y=mean_models_pos_shift, color='lightgreen', linestyle='--', linewidth=2)
    ax.axhline(y=mean_models_neg_shift, color='lightcoral', linestyle='--', linewidth=2)
    ax.bar(x[-1], humans_positive_shifts, width=bar_width, color="green")
    ax.bar(x[-1] + bar_width, humans_negative_shifts, width=bar_width, color="red")

    # Add text on bars
    for i, value in enumerate(all_models_positive_shifts):
        ax.text(x[i], value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=16)
    for i, value in enumerate(all_models_negative_shifts):
        ax.text(x[i] + bar_width, value + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=16)
    ax.text(x[-1], humans_positive_shifts + 1, f'{humans_positive_shifts:.1f}%', ha='center', va='bottom', fontsize=16)
    ax.text(x[-1] + bar_width, humans_negative_shifts + 1, f'{humans_negative_shifts:.1f}%', ha='center', va='bottom',
            fontsize=16)

    # Set x-ticks and labels
    ax.set_xticks(x + (bar_width / 2))
    ax.set_xticklabels(model_names + ['Humans'], fontsize=20, rotation=30, ha='right')
    ax.set_ylabel('Percentage of sentiment shifts', fontsize=20)
    ax.set_ylim(0, 90)
    ax.set_yticks(np.arange(0, 81, 10))
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=18, title="Base Sentiment", title_fontsize=18)
    plt.tight_layout()

    plt.savefig(os.path.join(dir_path, 'positive_negative_shifts.png'))

    plt.figure(figsize=(8, 6))
    all_models_diffs = []
    for i, model in enumerate(model_names):
        diff = all_model_results[i]['negative_flipped_minus_positive_flipped'] / 500 * 100
        all_models_diffs.append(diff)
    plt.barh(model_names, all_models_diffs, color='lightblue')
    # Adding text on bars
    for i, value in enumerate(all_models_diffs):
        offset = -1.3 if value < 0 else 1.3
        plt.text(value-offset, i, f'{value:.1f}', ha='center', va='center', fontsize=12)
    plt.xlabel('Negative base shifts minus positive base shifts (in %)', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, 'negative_minus_positive_shifts.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--human_annotations', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_analysis(args.models_dir, args.human_annotations, args.out_dir)
