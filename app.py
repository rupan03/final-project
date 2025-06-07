# app.py
import streamlit as st
import pickle
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load saved model and vectorizer
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

st.title("📱 Smartphone Review Sentiment Analyzer")
review = st.text_area("Enter your review:")

if st.button("Analyze"):
    cleaned = preprocess_text(review)
    vectorized = tfidf.transform([cleaned]).toarray()
    result = lr_model.predict(vectorized)[0]
    st.success(f"Predicted Sentiment: **{result}**")
