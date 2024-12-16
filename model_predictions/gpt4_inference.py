import sys

from openai import Client
import pandas as pd


def doSAGPT(baseSentence, firstSentnece, secondSentence, client):
    prompt = "Here's a sentence:\n" \
             "<sentence>.\n" \
             "Is the sentence Positive Or Negative? Write the answer as Json: " \
             "{'Sentiment' : 'Positive/Negative'}."
    basePrompt = prompt.replace("<sentence>", baseSentence)
    dic = [{"role": "user", "content": basePrompt}]
    chat = client.chat.completions.create(model="gpt-4-0613", messages=dic)
    baseReply = chat.choices[0].message.content.split("\n")

    firstPrompt = prompt.replace("<sentence>", firstSentnece)
    dic = [{"role": "user", "content": firstPrompt}]
    chat = client.chat.completions.create(model="gpt-4-0613", messages=dic)
    firstReply = chat.choices[0].message.content.split("\n")

    secondPrompt = prompt.replace("<sentence>", secondSentence)
    dic = [{"role": "user", "content": secondPrompt}]
    chat = client.chat.completions.create(model="gpt-4-0613", messages=dic)
    secondReply = chat.choices[0].message.content.split("\n")

    return baseReply, firstReply, secondReply


def makeTest(data):
    client = Client(api_key=sys.argv[1])
    for index, row in data.iterrows():
        if (index < 2024):
            continue
        print(index)
        processed_sentence = doSAGPT(row['sentence_text'], row['positive_framing'], row['negative_framing'], client)
        data.at[index, 'Base SA'] = processed_sentence[0][0]
        data.at[index, 'After Positive Framing SA'] = processed_sentence[1][0]
        data.at[index, 'After Negative Framing SA'] = processed_sentence[2][0]

        data.to_csv('combinedData_GPT_result.csv', index=False)


if __name__ == '__main__':
    data = pd.read_csv('combinedData_GPT_result.csv')
    data['Base SA'] = data['Base SA'].astype(str)
    data['After Positive Framing SA'] = data['After Positive Framing SA'].astype(str)
    data['After Negative Framing SA'] = data['After Negative Framing SA'].astype(str)
    makeTest(data)
