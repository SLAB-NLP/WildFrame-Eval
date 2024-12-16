import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from model_predictions.utils import *

models_list_ollama = [
    "gemma2:2b-instruct-fp16",
    "phi3.5:3.8b-mini-instruct-fp16",
    "qwen2.5:7b-instruct-fp16",
    "mistral:7b-instruct-v0.3-fp16",
    "llama3.1:8b-instruct-fp16",
    "gemma2:9b-instruct-fp16",
]


def inference_ollama(data_path, model_name, out_dir):
    llm = ChatOllama(model=model_name)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MSG,),
            ("human", USER_MSG,),
        ]
    )

    chain = prompt | llm

    data_df = pd.read_csv(data_path)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{model_name}_opposite_framing_predictions.csv')

    if os.path.exists(out_path):
        data_df = pd.read_csv(out_path)
    else:
        data_df = run_infer_ollama(data_df, chain, out_path)

    opposite_framing_pred = data_df['opposite_framing_raw_pred'].tolist()
    processed_out = process_preds(opposite_framing_pred)

    data_df['opposite_framing_processed_pred'] = processed_out
    data_df.to_csv(out_path, index=False)


def run_infer_ollama(data_df, chain, out_path):
    opposite_framing_pred = []
    prediction_num_tokens = []
    for i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        opposite_sentiment_framing = get_opposite_framing(row)
        answer = chain.invoke({"sentence": opposite_sentiment_framing})
        num_tokens = answer.usage_metadata['total_tokens']
        prediction_num_tokens.append(num_tokens)
        opposite_framing_pred.append(answer.content)
    print("Total tokens used:", sum(prediction_num_tokens))
    data_df['pred_num_tokens'] = prediction_num_tokens
    data_df['opposite_framing_raw_pred'] = opposite_framing_pred
    data_df.to_csv(out_path, index=False)
    return data_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to csv with the data to prompt the model with")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model to use for inference")
    parser.add_argument("--out_dir", type=str, default='model_predictions/inference',)

    args = parser.parse_args()
    inference_ollama(args.data_path, args.model_name, args.out_dir)
