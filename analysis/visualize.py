

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

    correlation_matrix = df_scores.iloc[:, 2:].corr()
    only_model_correlation = correlation_matrix[1:].drop(columns=['humans'])
    only_humans_correlation = correlation_matrix['humans'].drop(index=['humans'])


    # bar chart for humans
    colors = [MAP_MODELS_TO_COLORS[model] for model in only_model_correlation.index]
    fig, ax = plt.subplots(figsize=(6, 4))
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
    ax.set_xlabel("Pearson Coefficient")
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
             'total_became_neutral': total_became_neutral, 'total_kept': total_kept})

    plot_model_distribution(all_model_results, model_names, key='total', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_positive', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_negative', dir_path=out_dir)

    # humans distribution
    positive_humans = (group_by_original_sentiment.get_group('positive')['humans']*5).apply(round)
    flipped_positive = positive_humans.value_counts()
    negative_humans = (group_by_original_sentiment.get_group('negative')['humans']*5).apply(round)
    flipped_negative = negative_humans.value_counts()

    flipped_percentage = [[],[]]
    total_positive = len(df_scores[df_scores['base_sentiment'] == 'positive'])
    total_negative = len(df_scores[df_scores['base_sentiment'] == 'negative'])
    for i in range(6):
        flipped_percentage[0].append(round(flipped_positive[i] / total_positive * 100, 1))
        flipped_percentage[1].append(round(flipped_negative[i] / total_negative * 100, 1))
    for j in range(2):
        flipped_percentage[j].reverse()
    flipped_percentage = np.array(flipped_percentage)

    categories = [f"{i} Flipped" for i in range(6)]
    col_names = ['Positive', 'Negative']
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
    ax.set_xlabel('Percentage', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.10), ncol=len(categories), fontsize=11)
    plt.tight_layout()

    # Show the plot
    path = os.path.join(out_dir, f'humans_distribution.png')
    plt.savefig(path)




def get_model_name_for_print(file_name):
    name = file_name[:file_name.find('_')]
    name = name.lower()
    if 'fp16' in name:
        name = name[:name.find('-fp16')]
    name = name.replace('-instruct', '')
    name = name.replace('-it', '')
    name = name.replace('-mini', '')
    name = name.replace('-chat', '')
    name = name.replace('-llm', '')
    name = name.replace('-hf', '')
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
    categories = ['Flipped Sentiment', 'Became Neutral', 'Kept Original Sentiment']
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
    ax.set_xlabel('Percentage', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(categories), fontsize=12)
    plt.tight_layout()
    # Show the plot
    path = os.path.join(dir_path, f'{key}_models_distribution.png')
    plt.savefig(path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--human_annotations', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_analysis(args.models_dir, args.human_annotations, args.out_dir)
