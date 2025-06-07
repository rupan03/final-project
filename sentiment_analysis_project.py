{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "afecda55-854b-4c93-904e-a97db5aee894",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sentiment\n",
      "Positive    7719\n",
      "Negative    6858\n",
      "Neutral     2666\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Load the data (you already did this once)\n",
    "df = pd.read_csv('PROJECT_FINAL DRAFT.csv')\n",
    "\n",
    "# Drop rows with missing review\n",
    "df = df.dropna(subset=['review'])\n",
    "\n",
    "# Create Sentiment column\n",
    "# We'll map star ratings:\n",
    "# 1, 2 → Negative\n",
    "# 3 → Neutral\n",
    "# 4, 5 → Positive\n",
    "\n",
    "def get_sentiment(star):\n",
    "    if star <= 2:\n",
    "        return 'Negative'\n",
    "    elif star == 3:\n",
    "        return 'Neutral'\n",
    "    else:\n",
    "        return 'Positive'\n",
    "\n",
    "df['Sentiment'] = df['star'].apply(get_sentiment)\n",
    "\n",
    "# Check class distribution\n",
    "print(df['Sentiment'].value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "5fadefbc-fd82-44ae-bdb7-dabc745a8494",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\rupan\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>cleaned_review</th>\n",
       "      <th>Sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>update 15082020never give chance regret go ah...</td>\n",
       "      <td>update never give chance regret go aheadthe ic...</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>title obviously monsterand good performance</td>\n",
       "      <td>title obviously monsterand good performance</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>brilliant camera huge battery life brilliant ...</td>\n",
       "      <td>brilliant camera huge battery life brilliant d...</td>\n",
       "      <td>Positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>writing review using 6 daysi bought sumsung p...</td>\n",
       "      <td>writing review using daysi bought sumsung phon...</td>\n",
       "      <td>Neutral</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>defective product received gets 8 12 hours ch...</td>\n",
       "      <td>defective product received gets hours charging...</td>\n",
       "      <td>Negative</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review  \\\n",
       "0   update 15082020never give chance regret go ah...   \n",
       "1        title obviously monsterand good performance   \n",
       "2   brilliant camera huge battery life brilliant ...   \n",
       "3   writing review using 6 daysi bought sumsung p...   \n",
       "4   defective product received gets 8 12 hours ch...   \n",
       "\n",
       "                                      cleaned_review Sentiment  \n",
       "0  update never give chance regret go aheadthe ic...  Positive  \n",
       "1        title obviously monsterand good performance  Positive  \n",
       "2  brilliant camera huge battery life brilliant d...  Positive  \n",
       "3  writing review using daysi bought sumsung phon...   Neutral  \n",
       "4  defective product received gets hours charging...  Negative  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import re\n",
    "import string\n",
    "import nltk\n",
    "nltk.download('stopwords')\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "# Define preprocessing function\n",
    "def preprocess_text(text):\n",
    "    text = text.lower()  # Lowercase\n",
    "    text = re.sub(r'\\d+', '', text)  # Remove numbers\n",
    "    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation\n",
    "    text = re.sub(r'\\s+', ' ', text)  # Remove extra whitespace\n",
    "    stop_words = set(stopwords.words('english'))\n",
    "    text = ' '.join([word for word in text.split() if word not in stop_words])\n",
    "    return text\n",
    "\n",
    "# Apply to review column\n",
    "df['cleaned_review'] = df['review'].apply(preprocess_text)\n",
    "\n",
    "# Preview cleaned data\n",
    "df[['review', 'cleaned_review', 'Sentiment']].head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5808bae7-2b9a-4044-9e69-952f390c2a57",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(17243, 5000)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# Initialize TF-IDF Vectorizer\n",
    "tfidf = TfidfVectorizer(max_features=5000)\n",
    "\n",
    "# Fit and transform cleaned reviews\n",
    "X = tfidf.fit_transform(df['cleaned_review']).toarray()\n",
    "\n",
    "# Target variable\n",
    "y = df['Sentiment']\n",
    "\n",
    "# Check shape\n",
    "print(X.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fa213cde-8bce-4072-a050-66563e7fa441",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Split into train/test\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a14ac4d1-92f6-436a-83ae-224ddd33e921",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Results:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    Negative       0.92      0.97      0.95      1372\n",
      "     Neutral       0.91      0.78      0.84       533\n",
      "    Positive       0.99      1.00      0.99      1544\n",
      "\n",
      "    accuracy                           0.95      3449\n",
      "   macro avg       0.94      0.92      0.93      3449\n",
      "weighted avg       0.95      0.95      0.95      3449\n",
      "\n",
      "Accuracy: 0.9521600463902581\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, accuracy_score\n",
    "\n",
    "# Initialize model\n",
    "lr_model = LogisticRegression(max_iter=1000)\n",
    "\n",
    "# Train model\n",
    "lr_model.fit(X_train, y_train)\n",
    "\n",
    "# Predict\n",
    "y_pred_lr = lr_model.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Logistic Regression Results:\")\n",
    "print(classification_report(y_test, y_pred_lr))\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred_lr))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4a0cb766-67ed-4aae-bae4-1ba238fac503",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Naive Bayes Results:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    Negative       0.87      0.94      0.90      1372\n",
      "     Neutral       0.90      0.54      0.68       533\n",
      "    Positive       0.92      0.98      0.95      1544\n",
      "\n",
      "    accuracy                           0.90      3449\n",
      "   macro avg       0.90      0.82      0.84      3449\n",
      "weighted avg       0.90      0.90      0.89      3449\n",
      "\n",
      "Accuracy: 0.8970716149608582\n"
     ]
    }
   ],
   "source": [
    "from sklearn.naive_bayes import MultinomialNB\n",
    "\n",
    "# Initialize model\n",
    "nb_model = MultinomialNB()\n",
    "\n",
    "# Train model\n",
    "nb_model.fit(X_train, y_train)\n",
    "\n",
    "# Predict\n",
    "y_pred_nb = nb_model.predict(X_test)\n",
    "\n",
    "# Evaluation\n",
    "print(\"Naive Bayes Results:\")\n",
    "print(classification_report(y_test, y_pred_nb))\n",
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred_nb))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "bfded7c7-d356-4af9-9e1b-c0aa36b5d3ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "# Save Logistic Regression model\n",
    "with open('lr_model.pkl', 'wb') as f:\n",
    "    pickle.dump(lr_model, f)\n",
    "\n",
    "# Save TF-IDF vectorizer\n",
    "with open('tfidf.pkl', 'wb') as f:\n",
    "    pickle.dump(tfidf, f)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6e93cd9-3d69-409b-ab08-abf03a6bcce0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
