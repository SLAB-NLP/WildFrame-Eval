

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


def run_analysis(models_dir, human_annotations_path):
    all_models = {}
    for file_name in os.listdir(models_dir):
        all_models[file_name[:file_name.find('_')]] = pd.read_csv(os.path.join(models_dir, file_name))
    human_annotations = pd.read_csv(human_annotations_path)
    # all_models['human'] = human_annotations
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
            if original_sentiment == opposite_framing_sentiment:
                total_kept.append(row['ID'])
                if original_sentiment == 'positive':
                    orig_positive_kept.append(row['ID'])
                else:  # original_sentiment == 'negative'
                    orig_negative_kept.append(row['ID'])
            elif opposite_framing_sentiment == 'positive':  # original_sentiment == 'negative'
                total_flipped.append(row['ID'])
                orig_negative_flipped.append(row['ID'])
            elif opposite_framing_sentiment == 'negative':  # original_sentiment == 'positive'
                total_flipped.append(row['ID'])
                orig_positive_flipped.append(row['ID'])
            else: # opposite_framing_sentiment == 'neutral'
                total_became_neutral.append(row['ID'])
                if original_sentiment == 'positive':
                    orig_positive_became_neutral.append(row['ID'])
                else:  # original_sentiment == 'negative'
                    orig_negative_became_neutral.append(row['ID'])
        all_model_results.append(
            {'model': model, 'orig_positive_flipped': orig_positive_flipped,
             'orig_negative_flipped': orig_negative_flipped,
             'orig_positive_became_neutral': orig_positive_became_neutral,
             'orig_negative_became_neutral': orig_negative_became_neutral,
             'orig_positive_kept': orig_positive_kept,
             'orig_negative_kept': orig_negative_kept, 'total_flipped': total_flipped,
             'total_became_neutral': total_became_neutral, 'total_kept': total_kept})
    all_models_total = []
    for i, model in enumerate(model_names):
        total_flipped_for_plot = len(all_model_results[i]['total_flipped'])
        total_kept_for_plot = len(all_model_results[i]['total_kept'])
        total_to_neutral_for_plot = len(all_model_results[i]['total_became_neutral'])
        total_num_samples = total_kept_for_plot + total_to_neutral_for_plot + total_flipped_for_plot
        # make this as a percentage
        total_kept_for_plot = round(total_kept_for_plot/total_num_samples* 100, 1)
        total_to_neutral_for_plot = round(total_to_neutral_for_plot/total_num_samples* 100, 1)
        total_flipped_for_plot = round(100-total_kept_for_plot-total_to_neutral_for_plot, 1)
        assert total_kept_for_plot + total_to_neutral_for_plot + total_flipped_for_plot == 100
        all_models_total.append([total_flipped_for_plot, total_to_neutral_for_plot, total_kept_for_plot])
    all_models_total = np.array(all_models_total)
    categories = ['Flipped Sentiment', 'Became Neutral', 'Kept Original Sentiment']


    # Stacked bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = []
    bottoms = np.zeros(len(model_names))

    for i, category in enumerate(categories):
        bar = ax.bar(model_names, all_models_total[:, i], bottom=bottoms, label=category)
        bars.append(bar)

        # Adding text on bars
        for j, value in enumerate(all_models_total[:, i]):
            ax.text(j, bottoms[j] + value / 2, f'{value}%', ha='center', va='center', fontsize=10)

        bottoms += all_models_total[:, i]

    # Adding labels and title
    ax.set_ylabel('Percentage')
    ax.set_title('Model Comparison by Categories')
    plt.xticks(rotation=90)
    ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show the plot
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--human_annotations', type=str, required=True)
    args = parser.parse_args()

    run_analysis(args.models_dir, args.human_annotations)
