import os
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from model_predictions.utils import *

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference


models_list_watsonx = [
    "meta-llama/llama-3-1-70b-instruct"
    "mistralai/mistral-large"
]


def run_infer_wmv(data_df, model_name, out_path):
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com/",
        api_key=os.getenv("WATSONX_API_KEY")
    )

    client = APIClient(credentials)

    params = {
        "time_limit": 10000,
        "max_tokens": 100,
    }

    project_id = "abdcf218-2686-458c-baab-cf1e0a2a58c0"
    verify = False

    model = ModelInference(
        model_id=model_name,
        api_client=client,
        params=params,
        project_id=project_id,
        verify=verify,
    )

    opposite_framing_pred = []

    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        opposite_sentiment_framing = get_opposite_framing(row)
        current_prompt = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_MSG.format(sentence=opposite_sentiment_framing)}
        ]
        response = model.chat(messages=current_prompt)
        msg = response["choices"][0]["message"]["content"]
        opposite_framing_pred.append(msg)

    data_df['opposite_framing_raw_pred'] = opposite_framing_pred
    data_df.to_csv(out_path, index=False)

    return data_df


def inference_wmv(data_path, model_name, out_dir):

    data_df = pd.read_csv(data_path)

    os.makedirs(out_dir, exist_ok=True)
    model_name_for_fname = model_name.replace("/", "-")
    out_path = os.path.join(out_dir, f'{model_name_for_fname}_opposite_framing_predictions.csv')
    print(out_path)

    if os.path.exists(out_path):
        data_df = pd.read_csv(out_path)
    else:
        data_df = run_infer_wmv(data_df, model_name, out_path)

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
    parser.add_argument("--out_dir", type=str, default='model_predictions/inference',)

    args = parser.parse_args()
    inference_wmv(args.data_path, args.model_name, args.out_dir)


