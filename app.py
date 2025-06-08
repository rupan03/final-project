import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

# Fix NLTK stopwords error on Streamlit Cloud
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load model and vectorizer
lr_model = pickle.load(open('lr_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit app
st.title("📱 Smartphone Review Sentiment Analyzer")
review = st.text_area("Enter your review:")

from huggingface_hub import hf_hub_download
import pickle

# Download model
model_path = hf_hub_download(repo_id="mythic003/sentiment-analysis-model", filename="lr_model.pkl")
tfidf_path = hf_hub_download(repo_id="mythic003/sentiment-analysis-model", filename="tfidf.pkl")

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)
