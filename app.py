from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import pickle
import string
from nltk.stem import WordNetLemmatizer
app = Flask(__name__)

# Placeholder lists for inbox and spam emails
inbox = []
spam = []

# Load the tokenizer
with open(r"data\tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the threshold
with open(r"data\threshold.pickle", "rb") as handle:
    threshold = pickle.load(handle)

# Load the stopwords
with open(r"data\stopwords.pickle", "rb") as handle:
    stop_words = pickle.load(handle)

# Load the model
model = load_model("data/spam_detection_model.h5")

# Load the preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def remove_punctuations(text):
    # Remove punctuations
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocess_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_punctuations(text)
    text = remove_urls(text)
    text = lemmatization(text)
    return text

def classify_message(proba_spam, threshold):
    if proba_spam > threshold:
        return "Spam"
    else:
        return "Ham"

@app.route('/')
def home():
    return render_template('index.html', inbox=inbox, spam=spam)

@app.route('/send_email', methods=['POST'])
def send_email():
    email_content = request.form['email_content']
    preprocessed_content = preprocess_text(email_content)
    tokenized_content = tokenizer.texts_to_sequences([preprocessed_content])
    padded_content = pad_sequences(tokenized_content, maxlen=229, truncating='pre')

    # Assuming model is already defined and loaded
    predictions = model.predict(padded_content)
    proba_spam = predictions[0][1]
    result = classify_message(proba_spam, threshold)

    if result == 'Spam':
        spam.append(email_content)
    else:
        inbox.append(email_content)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
