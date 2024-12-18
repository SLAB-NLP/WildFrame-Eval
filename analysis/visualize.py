

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

company_names = ['google', 'mistralai', 'deepseek-ai']

# Function to determine color based on value
def get_color(value):
    norm = Normalize(vmin=0, vmax=7)
    return plt.cm.viridis(norm(6-value))

def run_analysis(models_dir, human_annotations_path, out_dir):
    all_models = {}
    for file_name in os.listdir(models_dir):
        name = file_name[:file_name.find('_')]
        if 'fp16' in name:
            name = name[:name.find('-fp16')]
        name = name.replace('-instruct', '')
        name = name.replace('-Instruct', '')
        name = name.replace('-it', '')
        name = name.replace('-mini', '')
        name = name.replace('-chat', '')
        name = name.replace(':', '-')
        for company in company_names:
            if company in name:
                name = name.replace(company+'-', '')
                break
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
    flipped_percentage = [[],[],[]]
    total_positive = len(np.concatenate(list(flipped_positive.values())))
    total_negative = len(np.concatenate(list(flipped_negative.values())))
    for i in range(6):
        flipped_percentage[0].append(round(len(flipped_total[i])/len(human_annotations)*100, 1))
        flipped_percentage[1].append(round(len(flipped_positive[i]) / total_positive * 100, 1))
        flipped_percentage[2].append(round(len(flipped_negative[i]) / total_negative * 100, 1))
    for j in range(3):
        flipped_percentage[j].reverse()
    flipped_percentage = np.array(flipped_percentage)

    categories = [f"{i} Flipped" for i in range(6)]
    col_names = ['Total', 'Positive', 'Negative']
    categories.reverse()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = []
    lefts = np.zeros(len(col_names))

    for i, category in enumerate(categories):
        color = get_color(i)

        bar = ax.barh(col_names, flipped_percentage[:, i], left=lefts, label=category, color=color, height=0.5)
        bars.append(bar)

        # Adding text on bars
        for j, value in enumerate(flipped_percentage[:, i]):
            if value == 0:
                continue
            ax.text(lefts[j] + value / 2, j, f'{value}%', ha='center', va='center', fontsize=10)

        lefts += flipped_percentage[:, i]

    # Adding labels and title
    ax.set_xlabel('Percentage')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    # Show the plot
    path = os.path.join(out_dir, f'humans_distribution.png')
    plt.savefig(path)

    plot_model_distribution(all_model_results, model_names, key='total', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_positive', dir_path=out_dir)
    plot_model_distribution(all_model_results, model_names, key='orig_negative', dir_path=out_dir)


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
        assert total_kept_for_plot + total_to_neutral_for_plot + total_flipped_for_plot == 100
        all_models_total.append([total_flipped_for_plot, total_to_neutral_for_plot, total_kept_for_plot])
    all_models_total = np.array(all_models_total)
    categories = ['Flipped Sentiment', 'Became Neutral', 'Kept Original Sentiment']
    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = []
    lefts = np.zeros(len(model_names))
    for i, category in enumerate(categories):
        norm = Normalize(vmin=0, vmax=7)
        value = (i * 2) + 1
        color = plt.cm.viridis(norm(6 - value))
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
