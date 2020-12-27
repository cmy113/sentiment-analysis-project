import re
import numpy as np
import pandas as pd
# Please run the nltk.download command to download the nltk data if you run locally!
# Heroku will download automatically using nltk.txt
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

'''
Preprocess the message before converting it into a matrix of token occurences (vectorizing)
1. Will do a sequence of text preprocessing
2. List of stop words to be used, we can either hardcode the stopwords in the code or use stopwords from nltk, 
remember to exclude words like 'not','nor' and etc as it will affect the meaning significantly! 
3. The stop words will be removed from the preprocessed message 
4. Message will then be lemmatized (e.g. ran, run, runs, running -> run) 
'''
def preprocess(text):
    # Preprocess the message
    text = text.lower()  # Convert everything to lowercase
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)  # Remove www, http://, https://
    text = re.sub('@[^\s]+', '', text)  # Remove @username
    text = re.sub('#([^\s]+)', '', text)  # Remove #hashtag
    text = re.sub('[^0-9A-Za-z \t]', ' ', text)  # Remove anything that is not alphanumeric, space or tab
    text = re.sub(r"(.)\1\1+", r"\1\1",
                  text)  # Replace heyyy into heyy or youuuu to youu, only keep the last 2 characters
    text = text.strip()  # remove trailing characters from both front and back

    # Create a list of stop words
    stop_words = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                  'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                  'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                  'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                  'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                  'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                  'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                  'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                  'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                  's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                  't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                  'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                  'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                  'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                  'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                  "youve", 'your', 'yours', 'yourself', 'yourselves']
    lemmatizer = WordNetLemmatizer()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)


'''

'''
def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform([preprocess(text)])
    sentiment = model.predict(textdata)
    score = round(np.amax(model.predict_proba(textdata)), 2)

    # Make a list of text with sentiment.
    data = [(text, sentiment, score)]

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment', 'score'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df
