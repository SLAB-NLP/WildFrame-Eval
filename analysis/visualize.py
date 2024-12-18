

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


def run_analysis(models_dir, human_annotations_path):
    all_models = {}
    for file_name in os.listdir(models_dir):
        all_models[file_name] = pd.read_csv(os.path.join(models_dir, file_name))
    human_annotations = pd.read_csv(human_annotations_path)
    all_models['human'] = human_annotations
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
        original_sentiment = all_models[model]['answer']
        opposite_framing_sentiment = all_models[model]['opposite_framing_processed_pred']
        if original_sentiment == opposite_framing_sentiment:
            total_kept.append(all_models[model]['ID'])
            if original_sentiment == 'positive':
                orig_positive_kept.append(all_models[model]['ID'])
            else:  # original_sentiment == 'negative'
                orig_negative_kept.append(all_models[model]['ID'])
        elif opposite_framing_sentiment == 'positive':  # original_sentiment == 'negative'
            total_flipped.append(all_models[model]['ID'])
            orig_negative_flipped.append(all_models[model]['ID'])
        elif opposite_framing_sentiment == 'negative':  # original_sentiment == 'positive'
            total_flipped.append(all_models[model]['ID'])
            orig_positive_flipped.append(all_models[model]['ID'])
        else: # opposite_framing_sentiment == 'neutral'
            total_became_neutral.append(all_models[model]['ID'])
            if original_sentiment == 'positive':
                orig_positive_became_neutral.append(all_models[model]['ID'])
            else:  # original_sentiment == 'negative'
                orig_negative_became_neutral.append(all_models[model]['ID'])
        all_model_results.append(
            {'model': model, 'orig_positive_flipped': orig_positive_flipped,
             'orig_negative_flipped': orig_negative_flipped,
             'orig_positive_became_neutral': orig_positive_became_neutral,
             'orig_negative_became_neutral': orig_negative_became_neutral,
             'orig_positive_kept': orig_positive_kept,
             'orig_negative_kept': orig_negative_kept, 'total_flipped': total_flipped,
             'total_became_neutral': total_became_neutral, 'total_kept': total_kept})
    total_flipped_for_plot = []
    total_kept_for_plot = []
    total_to_neutral_for_plot = []
    for i, model in enumerate(model_names):
        total_flipped_for_plot.append(len(all_model_results[i]['total_flipped']))
        total_kept_for_plot.append(len(all_model_results[i]['total_kept']))
        total_to_neutral_for_plot.append(len(all_model_results[i]['total_became_neutral']))
        total_num_samples = total_kept_for_plot[-1] + total_to_neutral_for_plot[-1] + total_flipped_for_plot[-1]
        # make this as a percentage
        total_kept_for_plot[-1] = round(total_kept_for_plot[-1]/total_num_samples, 2)
        total_to_neutral_for_plot[-1] = round(total_to_neutral_for_plot[-1]/total_num_samples, 2)
        total_flipped_for_plot[-1] = 1 - total_kept_for_plot[-1] - total_to_neutral_for_plot[-1]



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--human_annotations', type=str, required=True)
    args = parser.parse_args()
