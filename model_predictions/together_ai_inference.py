import os
import pandas as pd
from argparse import ArgumentParser
from together import Together
from tqdm import tqdm
from model_predictions.utils import *

models_list_together = [
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "deepseek-ai/deepseek-llm-67b-chat",
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3-8b-chat-hf",
    "meta-llama/Llama-3-70b-chat-hf",
]


def run_infer_together(data_df, model_name, out_path, allow_neutral):
    client = Together()

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


def inference_together(data_path, model_name, out_dir, allow_neutral):

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
        data_df = run_infer_together(data_df, model_name, out_path, allow_neutral)

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
    inference_together(args.data_path, args.model_name, args.out_dir, args.allow_neutral)


