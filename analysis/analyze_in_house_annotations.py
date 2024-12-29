import os
from argparse import ArgumentParser
import pandas as pd


def analyze_in_house_annotations(data_dir):
    total = 0
    total_negative = 0
    total_positive = 0
    total_not_suitable = 0
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
        print(f'Analyzing {filename}')
        with open(f'{data_dir}/{filename}', 'r') as f:
            df = pd.read_csv(f)
        only_labeled = df.dropna(subset=['answer'])
        total += len(only_labeled)
        total_negative += len(only_labeled[only_labeled["answer"] == "negative"])
        total_positive += len(only_labeled[only_labeled["answer"] == "positive"])
        total_not_suitable += len(only_labeled[only_labeled["answer"] == "not suitable"])
    print(f'Total: {total}')
    print(f'Negative: {total_negative}')
    print(f'Positive: {total_positive}')
    print(f'Not suitable: {total_not_suitable}')
    assert total == total_negative + total_positive + total_not_suitable


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/base_sentences')
    args = parser.parse_args()
    analyze_in_house_annotations(args.data_dir)
