import re

SYSTEM_MSG = "You are an advanced language model specializing in sentiment analysis. Your task is to analyze the sentiment of given input text and classify it as either 'Positive,' or 'Negative,' based on the overall tone and emotional content. Do not provide any additional information."
USER_MSG = "Here's a sentence:\n{sentence}\nIs the sentence Positive or Negative? Write the answer in a Json format: 'sentiment' : 'Positive' or 'Negative'."


def process_preds(opposite_framing_pred):
    processed_out = []
    for msg in opposite_framing_pred:
        match = re.search(r"{.*?}", msg, re.DOTALL)
        if match:
            dictionary_string = match.group(0)
            try:
                sentiment = eval(dictionary_string)['sentiment'].lower()
            except:
                if "mixed" in dictionary_string.lower():
                    sentiment = "Mixed"
                else:
                    sentiment = msg
        else:
            sentiment = msg
        processed_out.append(sentiment)
    return processed_out


def get_opposite_framing(row):
    sentiment = row['answer']
    if sentiment == 'positive':
        opposite_sentiment_framing = row['negative_framing']
    elif sentiment == 'negative':
        opposite_sentiment_framing = row['positive_framing']
    else:
        raise RuntimeError(f"Invalid sentiment: {sentiment}")
    return opposite_sentiment_framing
