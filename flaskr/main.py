import nltk
# Please run below command to download the nltk data if you run locally!
# Heroku will download automatically using nltk.txt
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
import re
import numpy as np
import pandas as pd

from flask import Flask

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='dev'
)
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

with open('./flaskr/static/vectoriser.pickle', 'rb') as input_file:
    vectorizer = pickle.load(input_file)

with open('./flaskr/static/Sentiment-LR.pickle', 'rb') as input_file:
    LRmodel = pickle.load(input_file)


@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        message = request.form['message']

        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))

    return render_template("base.html")


@app.route('/result', methods=('GET', 'POST'))
def result():
    message = session.get('message')
    df_pred = predict(vectorizer, LRmodel, message)
    sentiment = df_pred.head(1)['sentiment'].values[0]
    score = df_pred.head(1)['score'].values[0]
    if request.method == 'POST':
        message = request.form['message']

        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))

    return render_template("result.html", message=message, sentiment=sentiment, score=score)


def preprocess(text, stem=True):
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
    text = text.lower()  #  convert everything to lowercase
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)  # remove www, http://, https://
    text = re.sub('@[^\s]+', '', text)  # remove username
    text = re.sub('#([^\s]+)', '', text)  # remove # hashtag
    text = re.sub('[^0-9A-Za-z \t]', ' ', text)  # remove anything that is not alphanumeric, space or tab
    text = re.sub(r"(.)\1\1+", r"\1\1",
                  text)  # replace heyyy into heyy or youuuu to youu, only keep the last 2 characters
    text = text.strip()  # remove trailing characters

    tokens = []
    for token in text.split():
        #  Split the sentence into separate word
        #  Remove the stop word
        # Stem the word as well
        if token not in stop_words:
            if stem:
                tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)


def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform([preprocess(text)])
    sentiment = model.predict(textdata)
    score = round(np.amax(model.predict_proba(textdata)),2)

    # Make a list of text with sentiment.
    data = [(text, sentiment, score)]

    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment', 'score'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df
