
import os
import pandas as pd
from argparse import ArgumentParser
from openai import OpenAI
from tqdm import tqdm
from model_predictions.utils import *


models_list_openai = [
    "gpt-4o-2024-08-06",
]


def run_infer_openai(data_df, model_name, out_path, allow_neutral):
    client = OpenAI()

    opposite_framing_pred = []
    print(allow_neutral)

    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        opposite_sentiment_framing = get_opposite_framing(row)
        if allow_neutral:
            current_prompt = [
                {"role": "system", "content": SYSTEM_MSG_WITH_NEUTRAL},
                {"role": "user", "content": USER_MSG_WITH_NEUTRAL.format(sentence=opposite_sentiment_framing)}
            ]
        else:
            current_prompt = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_MSG.format(sentence=opposite_sentiment_framing)}
            ]
        response = client.chat.completions.create(
            model=model_name,
            messages=current_prompt,
        )
        opposite_framing_pred.append(response.choices[0].message.content)

    data_df['opposite_framing_raw_pred'] = opposite_framing_pred
    data_df.to_csv(out_path, index=False)

    return data_df


def inference_openai(data_path, model_name, out_dir, allow_neutral):

    data_df = pd.read_csv(data_path)

    os.makedirs(out_dir, exist_ok=True)
    model_name_for_fname = model_name.replace("/", "-")
    out_path = os.path.join(out_dir, f'{model_name_for_fname}_opposite_framing_predictions.csv')
    if allow_neutral:
        out_path = out_path.replace(".csv", "_with_neutral.csv")
    print(out_path)

    if os.path.exists(out_path):
        data_df = pd.read_csv(out_path)
    else:
        data_df = run_infer_openai(data_df, model_name, out_path, allow_neutral)

    opposite_framing_pred = data_df['opposite_framing_raw_pred'].tolist()
    processed_out = process_preds(opposite_framing_pred)

    data_df['opposite_framing_processed_pred'] = processed_out
    data_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to csv with the data to prompt the model with")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to use for inference")
    parser.add_argument("--allow_neutral", action='store_true', default=False,
                        help="Name of the model to use for inference")
    parser.add_argument("--out_dir", type=str, default='model_predictions/inference',)

    args = parser.parse_args()
    inference_openai(args.data_path, args.model_name, args.out_dir, args.allow_neutral)

