import pickle
import re
import numpy as np
import pandas as pd
from flask import Flask
# Please run the nltk.download command to download the nltk data if you run locally!
# Heroku will download automatically using nltk.txt
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


'''
Initiate a new flask app
1. Input some random secret key to be used by the application 
2. Input some flask commands that would be used by the application
'''
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='\xe0\xcd\xac#\x06\xd9\xe4\x00\xa5\xf2\x88\xc3\xef$\xa5\x05n\x97\xd8\x1269i\xd3'
)
from flask import (
    redirect, render_template, request, session, url_for
)


'''
Load the machine learning libraries 
1. Hashing vectorizer is used to transform the data into a matrix of token occurrences after text preprocessing
2. Logistic regression model is used to predict the sentiment on the newly computed matrix
'''
# Load the vectoriser and machine learning libraries
with open('./flaskr/static/HashingVectorizer.pickle', 'rb') as input_file:
    vectorizer = pickle.load(input_file)
with open('./flaskr/static/LogisticRegression.pickle', 'rb') as input_file:
    model = pickle.load(input_file)


'''
Home Page
1. It will take both GET and POST requests 
2. For GET request, base.html (homepage) will be rendered without any results shown
3. For POST request, input message will be obtained from the form in base.html.
    a) Session will then be cleared (to remove anything belonged to previous session) and 'message' will be passed into the session 
    so that it can be reused throughout the session
    b) The page will then be redirected to /result page
'''
@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        message = request.form['message']

        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))

    return render_template("base.html")

'''
Result Page
1. It will take both GET and POST requests 
2. For GET request, 'message' will be obtained from the session, remember the 'message' is from the Home page! 
    a) Sentiment and its score(probability) will be predicted by passing in the vectorizer, model and message from the session
    b) The result page will then be rendered based on the message, sentiment and score computed by the predictions
3. For POST request, input message will be obtained from the form in result.html 
    a) Session will then be cleared (to remove anything belonged to previous session) and 'message' will be passed into the session 
    so that it can be reused throughout the session
    c) The page will then be redirected to /result page
'''
@app.route('/result', methods=('GET', 'POST'))
def result():
    message = session.get('message')
    df_pred = predict(vectorizer, model, message)
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
