import os
from argparse import ArgumentParser
import pandas as pd


def combine_batches(data_dir):
    combined_data = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        data = pd.read_csv(os.path.join(data_dir, fname))
        combined_data.append(data)
    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df


def process_scores(data_dir, output_path):
    combined_df = combine_batches(data_dir)
    combined_df['Answer.sentiment.label'] = combined_df['Answer.sentiment.label'].apply(lambda x: x.lower())
    group_by_sentence = combined_df.groupby('Input.ID')
    out_rows = []
    for sentence_id in group_by_sentence.groups:
        only_sentence_data = group_by_sentence.get_group(sentence_id)
        base_sentence_text = only_sentence_data['Input.sentence_text'].iloc[0]
        base_sentiment = only_sentence_data["Input.answer"].iloc[0]
        opposite_framing_sentence = only_sentence_data['Input.framed_sentence'].iloc[0]
        count = only_sentence_data['Answer.sentiment.label'].value_counts()
        positive_count = count.get('positive', 0)
        negative_count = count.get('negative', 0)
        positive_score = positive_count / (positive_count + negative_count)
        negative_score = 1 - positive_score
        majority_sentiment = 'positive' if positive_score > negative_score else 'negative'
        majority_confidence = max(positive_score, negative_score)
        row = {
            'sentence_id': sentence_id,
            'base_sentence_text': base_sentence_text.strip(), 'base_sentiment': base_sentiment,
            'opposite_framing_sentence': opposite_framing_sentence.strip(),
            'positive_score': positive_score, 'negative_score': negative_score,
            'majority_sentiment': majority_sentiment, 'majority_confidence': majority_confidence
        }
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values('sentence_id')
    out_df["sentiment_flip_after_framing"] = out_df["base_sentiment"] != out_df["majority_sentiment"]
    print(out_df["sentiment_flip_after_framing"].sum())
    print(out_df.loc[out_df["sentiment_flip_after_framing"]]["majority_confidence"].mean())
    print(out_df.loc[~out_df["sentiment_flip_after_framing"]]["majority_confidence"].mean())
    out_df.to_csv(output_path, index=False)

    analyze_turkers(combined_df, out_df, output_path)


def analyze_turkers(combined_df, out_df, output_path):
    group_by_user = combined_df.groupby('WorkerId')
    majority_sentiment_all = []
    for i, row in combined_df.iterrows():
        sentence_id = row['Input.ID']
        majority_sentiment = out_df.loc[out_df['sentence_id'] == sentence_id]['majority_sentiment'].iloc[0]
        majority_sentiment_all.append(majority_sentiment)
    combined_df['majority_sentiment'] = majority_sentiment_all
    worker_stats = []
    for worker_id in group_by_user.groups:
        only_user_annotations = group_by_user.get_group(worker_id)
        count = only_user_annotations['Answer.sentiment.label'].value_counts()
        positive_count = count.get('positive', 0)
        negative_count = count.get('negative', 0)
        num_annotations = positive_count + negative_count
        flipped_sentiment = only_user_annotations['Answer.sentiment.label'] != only_user_annotations['Input.answer']
        voted_like_majority = only_user_annotations['Answer.sentiment.label'] == only_user_annotations['majority_sentiment']
        row = {"worker_id": worker_id,
               "num_annotations": num_annotations,
               "positive_count": round(positive_count / num_annotations,3),
               "negative_count": round(negative_count / num_annotations,3),
               "flipped_sentiment": round(flipped_sentiment.sum() / num_annotations,3),
               "voted_like_majority": round(voted_like_majority.sum() / num_annotations,3)}
        print(row)
        worker_stats.append(row)
    worker_stats_df = pd.DataFrame(worker_stats)
    path = os.path.dirname(output_path)
    path = os.path.join(path, "worker_stats.csv")
    worker_stats_df.to_csv(path, index=False)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing the result batches from mturk")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output csv file")
    args = parser.parse_args()
    process_scores(args.data_dir, args.output_path)
