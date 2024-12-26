
from flair.data import Sentence
from flair.nn import Classifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, ttest_rel
from tqdm import tqdm
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# sentence_text,answer,positive_framing,negative_framing
def get_flair_score(data_df, out_dir):
    classifier = Classifier.load('en-sentiment')
    group_by_sentiment = data_df.groupby('answer')
    # Prepare data for plotting
    plot_data = []

    for sentiment in ['negative', 'positive']:
        pos_score_orig = []
        pos_score_opposite_framing = []
        df_sentiment = group_by_sentiment.get_group(sentiment)
        opposite_framing = 'positive_framing' if sentiment == 'negative' else 'negative_framing'

        for row in tqdm(df_sentiment.itertuples(), total=len(df_sentiment)):
            orig_sentence = Sentence(row.sentence_text)
            opposite_sentence = Sentence(getattr(row, opposite_framing))
            classifier.predict(orig_sentence)
            classifier.predict(opposite_sentence)

            orig_pos_score = orig_sentence.labels[0].score
            if orig_sentence.labels[0].value.lower() == 'negative':
                orig_pos_score = 1 - orig_sentence.labels[0].score

            after_opposite_framing_pos_score = opposite_sentence.labels[0].score
            if opposite_sentence.labels[0].value.lower() == 'negative':
                after_opposite_framing_pos_score = 1 - opposite_sentence.labels[0].score

            pos_score_orig.append(orig_pos_score)
            pos_score_opposite_framing.append(after_opposite_framing_pos_score)

        # Add data to plot_data
        plot_data.extend([
            {'Sentiment': sentiment.capitalize(), 'Type': 'Before', 'Score': score}
            for score in pos_score_orig
        ])
        plot_data.extend([
            {'Sentiment': sentiment.capitalize(), 'Type': 'After', 'Score': score}
            for score in pos_score_opposite_framing
        ])

    out_path = f'{out_dir}/flair_score_before_after_framing'
    plot_violin(plot_data, out_path)


def plot_violin(plot_data, out_path):
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    # Create the violin plot
    plt.figure()
    sns.violinplot(
        data=plot_df,
        x='Sentiment',
        y='Score',
        hue='Type',
        split=True,  # For side-by-side comparison within each category
        palette='Set2'
    )
    plt.xlabel('Base Sentence Sentiment', fontsize=14)
    plt.ylabel('Positive Sentiment Score', fontsize=14)
    plt.legend(title='Framing', loc='upper left', fontsize=12)
    plt.savefig(out_path+'.png')

# Define a function to calculate positive sentiment score
def calculate_pos_score(model, tokenizer, text, original_sentiment):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    pos_score = probabilities[:, 2]
    if original_sentiment == 'negative':
        pos_score = pos_score + probabilities[:, 1]
    return pos_score.numpy()

def get_roberta_score(data_df, out_dir):

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    group_by_sentiment = data_df.groupby('answer')
    plot_data = []

    for sentiment in ['negative', 'positive']:

        df_sentiment = group_by_sentiment.get_group(sentiment)
        opposite_framing = 'positive_framing' if sentiment == 'negative' else 'negative_framing'
        pos_score_orig = calculate_pos_score(model, tokenizer, df_sentiment.sentence_text.to_list(), sentiment)
        pos_score_opposite_framing = calculate_pos_score(model, tokenizer, df_sentiment[opposite_framing].to_list(), sentiment)

        # Add data to plot_data
        plot_data.extend([
            {'Sentiment': sentiment.capitalize(), 'Type': 'Before', 'Score': score}
            for score in pos_score_orig
        ])
        plot_data.extend([
            {'Sentiment': sentiment.capitalize(), 'Type': 'After', 'Score': score}
            for score in pos_score_opposite_framing
        ])

    out_path = f'{out_dir}/roberta_score_before_after_framing'
    plot_violin(plot_data, out_path)





if __name__ == '__main__':
    data = pd.read_csv('data/with_framing/data_with_framing.csv')
    out_dir = '_output'
    get_roberta_score(data, out_dir)
    get_flair_score(data, out_dir)
