

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
from scipy.spatial.distance import cdist

company_names = ['google', 'mistralai', 'deepseek-ai', 'meta-llama']



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



# Function to determine color based on value
def get_color(value):
    norm = Normalize(vmin=0, vmax=8)
    return plt.cm.viridis(norm(7-value))

def mean_absolute_distance(v1, v2):
    return np.mean(np.abs(v1 - v2)).round(2)


def run_analysis(models_dir, human_annotations_path, out_dir):
    df_scores = generate_labels_csv(models_dir, human_annotations_path)
    pairwise_cosine_similarity = cosine_similarity(df_scores.iloc[:, 2:].T)
    pairwise_cosine_similarity = np.round(pairwise_cosine_similarity, 2)


    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(pairwise_cosine_similarity, cmap="Reds", aspect="auto", vmin=0.5, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.set_label("Cosine Similarity")

    ax.set_xticks(range(len(df_scores.columns[2:])), df_scores.columns[2:], rotation=90)
    ax.set_yticks(range(len(df_scores.columns[2:])), df_scores.columns[2:])
    ax.set_title("Pairwise Cosine Similarity")
    # Add annotations (text on the heatmap)
    for i in range(len(pairwise_cosine_similarity)):
        for j in range(len(pairwise_cosine_similarity)):
            ax.text(j, i, f'{pairwise_cosine_similarity[i, j]:.2f}',
                    ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairwise_cosine_similarity.png'))

    correlation_matrix = df_scores.iloc[:, 2:].corr()
    only_model_correlation = correlation_matrix[1:].drop(columns=['humans'])
    only_humans_correlation = correlation_matrix['humans'].drop(index=['humans'])
    models_with_colors = {
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
    colors = [models_with_colors[model] for model in only_model_correlation.index]
    # bar chart for humans
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(only_humans_correlation.index, only_humans_correlation.values, color=colors)
    # set x lim to 1
    # ax.set_xlim(0, 1)
    ax.set_xlabel("Pearson Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_with_humans.png'))
    masked_data = np.tril(only_model_correlation)
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(masked_data, cmap="Reds", aspect="auto", vmin=0, vmax=1, interpolation='none',)
    cbar = fig.colorbar(cax)
    cbar.set_label("Pearson Coefficient")
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(range(len(only_model_correlation)), only_model_correlation.columns, rotation=90)
    ax.set_yticks(range(len(only_model_correlation)), only_model_correlation.columns)
    # Add annotations (text on the heatmap)
    for i in range(len(only_model_correlation)):
        for j in range(len(only_model_correlation)):
            if i < j:
                continue
            ax.text(j, i, f'{only_model_correlation.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairwise_correlation_matrix.png'))

    only_models = df_scores.iloc[:, 2:].T
    vectors = only_models.values
    # Compute pairwise mean absolute distances
    num_vectors = len(vectors)
    mad_matrix = np.zeros((num_vectors, num_vectors))

    for i in range(num_vectors):
        for j in range(num_vectors):
            mad_matrix[i, j] = mean_absolute_distance(vectors[i], vectors[j])

    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(mad_matrix, cmap="Reds_r", aspect="auto", vmin=0, vmax=0.5)
    cbar = fig.colorbar(cax)
    cbar.set_label("Absolute Difference")
    cbar.ax.invert_yaxis()

    ax.set_xticks(range(len(df_scores.columns[2:])), df_scores.columns[2:], rotation=90)
    ax.set_yticks(range(len(df_scores.columns[2:])), df_scores.columns[2:])
    ax.set_title("Pairwise Absolute Difference")
    # Add annotations (text on the heatmap)
    for i in range(len(mad_matrix)):
        for j in range(len(mad_matrix)):
            ax.text(j, i, f'{mad_matrix[i, j]:.2f}',
                    ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pairwise_absolute_diff.png'))

    all_models = {}
    for file_name in os.listdir(models_dir):
        name = get_model_name_for_print(file_name)
        all_models[name] = pd.read_csv(os.path.join(models_dir, file_name))
    human_annotations = pd.read_csv(human_annotations_path)
    model_names = sorted(list(all_models.keys()))
    all_model_results = []
    for model in model_names:
        orig_positive_flipped = []
        orig_negative_flipped = []
        orig_positive_became_neutral = []
        orig_negative_became_neutral = []
        orig_positive_kept = []
        orig_negative_kept = []
        total_flipped = []
        total_became_neutral = []
        total_kept = []
        for i, row in all_models[model].iterrows():
            original_sentiment = row['answer']
            opposite_framing_sentiment = row['opposite_framing_processed_pred']
            sentence_id = row['ID']
            if original_sentiment == opposite_framing_sentiment:
                total_kept.append(sentence_id)
                if original_sentiment == 'positive':
                    orig_positive_kept.append(sentence_id)
                else:  # original_sentiment == 'negative'
                    orig_negative_kept.append(sentence_id)
            elif opposite_framing_sentiment == 'positive':  # original_sentiment == 'negative'
                total_flipped.append(sentence_id)
                orig_negative_flipped.append(sentence_id)
            elif opposite_framing_sentiment == 'negative':  # original_sentiment == 'positive'
                total_flipped.append(sentence_id)
                orig_positive_flipped.append(sentence_id)
            else: # opposite_framing_sentiment == 'neutral'
                total_became_neutral.append(sentence_id)
                if original_sentiment == 'positive':
                    orig_positive_became_neutral.append(sentence_id)
                else:  # original_sentiment == 'negative'
                    orig_negative_became_neutral.append(sentence_id)
        all_model_results.append(
            {'model': model, 'orig_positive_flipped': orig_positive_flipped,
             'orig_negative_flipped': orig_negative_flipped,
             'orig_positive_became_neutral': orig_positive_became_neutral,
             'orig_negative_became_neutral': orig_negative_became_neutral,
             'orig_positive_kept': orig_positive_kept,
             'orig_negative_kept': orig_negative_kept, 'total_flipped': total_flipped,
             'total_became_neutral': total_became_neutral, 'total_kept': total_kept})
    flipped_total = {i: [] for i in range(6)}
    flipped_positive = {i: [] for i in range(6)}
    flipped_negative = {i: [] for i in range(6)}

    # todo for each sentence id, what did the models label? 0 is kept the same, 1 is flipped, 2 is became neutral
    # num_models = predictions.shape[1]
    #
    # # Calculate agreement matrix
    # agreement_matrix = np.zeros((num_models, num_models))
    #
    # for i in range(num_models):
    #     for j in range(num_models):
    #         agreement = np.mean(predictions[:, i] == predictions[:, j])
    #         agreement_matrix[i, j] = agreement
    #
    # # Plot heatmap
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(agreement_matrix, annot=True, fmt=".2f", cmap="Blues",
    #             xticklabels=[f'Model {i + 1}' for i in range(num_models)],
    #             yticklabels=[f'Model {i + 1}' for i in range(num_models)])
    # plt.title("Model Agreement Heatmap")
    # plt.xlabel("Models")
    # plt.ylabel("Models")
    # plt.tight_layout()
    # plt.show()


    for i, row in human_annotations.iterrows():
        number_voted_majority = int(row['majority_confidence'] * 5)
        if row['sentiment_flip_after_framing']:
            flipped_total[number_voted_majority].append(row['sentence_id'])
            if row['base_sentiment'] == 'positive':
                flipped_positive[number_voted_majority].append(row['sentence_id'])
            else:  # row['base_sentiment'] == 'negative'
                flipped_negative[number_voted_majority].append(row['sentence_id'])
        else:
            flipped_total[5-number_voted_majority].append(row['sentence_id'])
            if row['base_sentiment'] == 'positive':
                flipped_positive[5-number_voted_majority].append(row['sentence_id'])
            else:  # row['base_sentiment'] == 'negative'
                flipped_negative[5-number_voted_majority].append(row['sentence_id'])
    flipped_percentage = [[],[]]
    total_positive = len(np.concatenate(list(flipped_positive.values())))
    total_negative = len(np.concatenate(list(flipped_negative.values())))
    for i in range(6):
        # flipped_percentage[0].append(round(len(flipped_total[i])/len(human_annotations)*100, 1))
        flipped_percentage[0].append(round(len(flipped_positive[i]) / total_positive * 100, 1))
        flipped_percentage[1].append(round(len(flipped_negative[i]) / total_negative * 100, 1))
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
            ax.text(lefts[j] + value / 2, j, f'{value}%', ha='center', va='center', fontsize=10)

        lefts += flipped_percentage[:, i]

    # Adding labels and title
    ax.set_xlabel('Percentage')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.10), ncol=len(categories))
    plt.tight_layout()

    # Show the plot
    path = os.path.join(out_dir, f'humans_distribution.png')
    plt.savefig(path)

    plot_model_distribution(all_model_results, model_names, key='total', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_positive', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_negative', dir_path=out_dir)


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
    ax.set_xlabel('Percentage')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=len(categories))
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
