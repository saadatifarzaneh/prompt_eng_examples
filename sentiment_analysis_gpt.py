import pandas as pd
import time
from openai import OpenAI

def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

OPENAI_API_KEY = read_api_key('openai_api_key.txt')
client = OpenAI(api_key=OPENAI_API_KEY)

def get_sentiment(comment, chatGPT_ver= "gpt-3.5-turbo"):
    
    response = client.chat.completions.create(model=chatGPT_ver,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that classifies the sentiment of comments."},
        {"role": "user", "content": f"Classify the sentiment of this comment: \"{comment}\". Respond with -1 for negative, 0 for neutral, and 1 for positive. Please don't answer anything but those numbers."}
    ],
    temperature=0,
    max_tokens=1000)

    sentiment = response.choices[0].message.content.strip()
    try:
        return int(sentiment)
    except ValueError:
        return None

def calculate_accuracy(predictions, actual):
    correct = sum([1 for pred, act in zip(predictions, actual) if pred == act])
    return correct / len(actual)

def calculate_sentiments(input="test_limit-updated.csv", max_rows=10, gpt_ver="gpt-3.5-turbo"):
    predictions = []
    big_mistakes = 0
    df = pd.read_csv(input, nrows=max_rows)

    for index, row in df.iterrows():
        comment = row['comment']
        annotated_sentiment = row['sentiment']
        predicted_sentiment = get_sentiment(comment)

        while predicted_sentiment is None:
            print("Sleep for 60 seconds and retrying...")
            time.sleep(60)
            predicted_sentiment = get_sentiment(comment, chatGPT_ver=gpt_ver)

        predictions.append(predicted_sentiment)

        if (annotated_sentiment == -1 and predicted_sentiment == 1) or (annotated_sentiment == 1 and predicted_sentiment == -1):
            big_mistakes += 1

        print(f"comment {index+1}/{len(df)}:\t Expected: {annotated_sentiment}, Predicted: {predicted_sentiment}")

    accuracy = calculate_accuracy(predictions, df['sentiment'])
    return accuracy, big_mistakes
 
# MAIN PART
max_rows = 15    
accuracy, big_mistakes = calculate_sentiments("test_limit-updated.csv", max_rows=max_rows, gpt_ver="gpt-3.5-turbo")
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Number of big mistakes (negative recognized as positive and vice versa): {big_mistakes}")
