import uvicorn
import joblib
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import ydf
from sklearn.preprocessing import StandardScaler

import re
import string
import nltk
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

from fastapi import FastAPI, HTTPException, Depends, status, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os
from pydantic import BaseModel
from typing import List

class EnhancedSMSSpamFilter:
    """
    Enhanced SMS Spam Filter with TF-IDF, Sentiment Analysis, and Spellchecking
    """
    def __init__(self, tfidf_features=3000):

        self.model = None

        self.tfidf_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=tfidf_features,
            output_mode="tf_idf",
            ngrams=1,
            standardize=self.custom_standardization
        )
        self.scaler = StandardScaler()
        self.sia = SentimentIntensityAnalyzer()

        # Feature names for later analysis
        self.feature_names = []

        with open('gubat_model_configs/stopwords-tl.json', 'r') as f:
            tl_stopwords = json.load(f)

        with open('gubat_model_configs/stopwords-en.json', 'r') as f:
            eng_stopwords = json.load(f)

        self.all_stopwords_set = set(tl_stopwords + eng_stopwords)

        # Spam indicators
        self.spam_words = ['free', 'win', 'winner', 'cash', 'prize', 'claim',
                           'urgent', 'offer', 'guaranteed', 'credit', 'loan',
                           'congrats', 'congratulations', 'click', 'buy', 'now',
                           'limited', 'act', 'call', 'text', 'reply', 'stop',
                           'libre', 'panalo', 'manalo', 'nanalo', 'premyo', 'swerte',
                           'click here', 'napanalunan', 'paclaim', 'xyz', 'spin',
                           'lucky', 'redeem', 'bonuses', 'vip', 'reward', 'jackpot',
                            'b0nus', 'cH@NC3', 'b0nu5', 'lph0ne', 's@s@kyan', 'f@cebook',
                           'mess3nger', 'nanal0', 'mess@ge', 'sc@tter', 'm3ssag3', 'unl!',
                           'pumaldo', 'paldo', 'm3ssage', 'm3ssager', 'sir/maam'
                          ]

        self.money_words = ['$', 'dollar', 'pound', 'cash', 'money',
                           'price', 'cost', 'fee', 'cheap', 'free', 'prize',
                           'p', 'php', 'Ã¢', 'Â±', 'pera', 'bonus', 'bayad',
                            'diskwento', 'libre', 'peso'
                           ]

        self.urgency_words = ['urgent', 'hurry', 'now', 'immediately', 'instant',
                            'act', 'limited', 'expires', 'deadline', 'today',
                            'ngayon', 'confirm today', 'within', 'ngayun', 'ngaun',
                            'n0w'
                            ]

    def custom_standardization(self, input_data):
        text = tf.strings.lower(input_data)

        text = tf.strings.regex_replace(text, r'\bunl!\b', 'unli')

        leet_char_map = {
            '0': 'o',
            '@': 'a',
            '3': 'e',
            '5': 's',
            '1': 'i',
            '$': 's',
            '!': 'i',
            '6': 'g',
            '4': 'a'
        }

        for char, replacement in leet_char_map.items():
            # Must escape the char in case it's a regex special char (like '$')
            safe_char = re.escape(char)

            # Logic 1: "Enclosed by letter"
            # Pattern: ([a-z]) + CHAR + ([a-z])
            # Replace: \1 (group 1) + REPLACEMENT + \2 (group 2)
            # Example: m3s -> mes, lph0n -> lphon
            enclosed_pattern = r'([a-z])' + safe_char + r'([a-z])'
            replacement_pattern = r'\1' + replacement + r'\2'
            text = tf.strings.regex_replace(text, enclosed_pattern, replacement_pattern)

            # Logic 2: "Ends with a number/symbol"
            # Pattern: ([a-z]) + CHAR + \b (word boundary)
            # Replace: \1 (group 1) + REPLACEMENT
            # Example: b0nu5 -> bonus, nanal0 -> nanalo
            ends_with_pattern = r'([a-z])' + safe_char + r'\b'
            replacement_pattern = r'\1' + replacement
            text = tf.strings.regex_replace(text, ends_with_pattern, replacement_pattern)

            # (Optional) Logic 3: "Starts with a number/symbol"
            # Pattern: \b (word boundary) + CHAR + ([a-z])
            # Replace: REPLACEMENT + \1 (group 1)
            # Example: f4cebook -> facebook
            starts_with_pattern = r'\b' + safe_char + r'([a-z])'
            replacement_pattern = replacement + r'\1'
            text = tf.strings.regex_replace(text, starts_with_pattern, replacement_pattern)

        # Now, strip all remaining punctuation
        lowercased_and_stripped = tf.strings.regex_replace(text, '[%s]' % re.escape(string.punctuation), '')

        # Continue with stopword removal
        stopword_pattern = r'\b(' + '|'.join(re.escape(s) for s in self.all_stopwords_set) + r')\b'
        no_stopwords = tf.strings.regex_replace(lowercased_and_stripped, stopword_pattern, ' ')

        # Clean up extra whitespace
        clean_text = tf.strings.regex_replace(no_stopwords, r'\s+', ' ')

        return clean_text


    def extract_base_features(self, messages):
        """
        Extract comprehensive handcrafted features from SMS messages.
        """
        features = pd.DataFrame()

        features['has_dti_code'] = messages.str.contains(r'DTI\d{6}', case=False,
        regex=True).astype(int)

        features['has_ref_no'] = messages.str.contains(
        r'ref\.?\s?no\.?\s?(\d{13}|[0-9a-f]{12})', case=False, regex=True).astype(int)

        # Basic text statistics
        features['msg_length'] = messages.str.len()
        features['word_count'] = messages.str.split().str.len()
        features['avg_word_length'] = messages.apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )

        features['num_digits'] = messages.str.findall(r'\d').str.len()
        features['digit_ratio'] = features['num_digits'] / (features['msg_length'] + 1)
        features['num_spaces'] = messages.str.count(' ')
        features['space_ratio'] = features['num_spaces'] / (features['msg_length'] + 1)

        # Punctuation features
        features['total_punctuation'] = messages.apply(
            lambda x: sum(1 for char in x if char in string.punctuation)
        )
        features['punctuation_ratio'] = features['total_punctuation'] / (features['msg_length'] + 1)

        # Special character features
        features['num_special_chars'] = messages.apply(
            lambda x: sum(1 for char in x if not char.isalnum() and not char.isspace())
        )
        features['special_char_ratio'] = features['num_special_chars'] / (features['msg_length'] + 1)

        # Money patterns
        features['has_money_pattern'] = messages.str.contains(
            r'\b(p\d+k?|\d+[pk])\b', case=False, regex=True
        ).astype(int)
        features['num_money_patterns'] = messages.str.findall(
            r'\b(p\d+k?|\d+[pk])\b'
        ).str.len()

        # Shorten URL detection
        features['has_url'] = messages.str.contains(
            r'http[s]?://|www\.|bit\.ly|tinyurl', case=False, regex=True
        ).astype(int)

        features['has_phone'] = messages.str.contains(
          r'\b\d{10,15}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', regex=True
        ).astype(int)

        features['has_email'] = messages.str.contains(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True
        ).astype(int)

        # Spam keyword features
        features['spam_word_count'] = messages.apply(
            lambda x: sum(1 for word in x.lower().split() if word in self.spam_words)
        )
        features['spam_word_ratio'] = features['spam_word_count'] / (features['word_count'] + 1)

        # Money-related features
        features['money_mentions'] = messages.apply(
            lambda x: sum(1 for word in self.money_words if word.lower() in x.lower())
        )
        features['urgency_score'] = messages.apply(
            lambda x: sum(1 for word in x.lower().split() if word in self.urgency_words)
        )

        features['has_price'] = messages.str.contains(
            r'\$\d+|p\d+|php\d+|\d+php\|\d+p|\d+k|\d+\s*(dollar|pound|euro|usd|gbp|peso|php|p|k)', case=False, regex=True
        ).astype(int)

        # Consecutive capitals and repeating characters
        features['has_repeating_chars'] = messages.str.contains(
            r'(.)\1{2,}', regex=True
        ).astype(int)

        # Call-to-action detection
        features['has_cta'] = messages.str.contains(
            r'\b(call|text|reply|click|visit|stop|subscribe|unsubscribe|confirm|msg|message|messge|msge)\b',
            case=False, regex=True
        ).astype(int)

        # Alphanumeric features
        features['num_alphanumeric'] = messages.apply(
            lambda x: sum(1 for char in x if char.isalnum())
        )
        features['alphanumeric_ratio'] = features['num_alphanumeric'] / (features['msg_length'] + 1)

        # Sentence count
        features['sentence_count'] = messages.apply(
            lambda x: len(re.findall(r'[.!?]+', x)) + 1
        )

        # Average sentence length
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']

        # Time pressure indicators
        features['has_time_limit'] = messages.str.contains(
            r'\b(\d+\s*(hours?|hrs?|minutes?|mins?|days?|weeks?))|today|tonight|tomorrow\b',
            case=False, regex=True
        ).astype(int)

        features['has_sms_code'] = messages.str.contains(
            r'\b[A-Z]{4,10}\b\s+to\s+\d{4,6}', regex=True
        ).astype(int)

        return features

    def extract_sentiment_features(self, messages):
        """
        Extract sentiment features using TextBlob and VADER
        """
        features = pd.DataFrame()

        # VADER sentiment analysis
        vader_scores = messages.apply(lambda x: self.sia.polarity_scores(x))
        features['vader_positive'] = vader_scores.apply(lambda x: x['pos'])
        features['vader_negative'] = vader_scores.apply(lambda x: x['neg'])
        features['vader_neutral'] = vader_scores.apply(lambda x: x['neu'])
        features['vader_compound'] = vader_scores.apply(lambda x: x['compound'])

        # Emotional intensity features
        features['emotion_range'] = features['vader_positive'] - features['vader_negative']
        features['emotion_intensity'] = features['vader_positive'] + features['vader_negative']

        return features

    def extract_tfidf_features(self, messages, fit=False):
        """
        Extract TF-IDF features using TextVectorization
        """
        # Ensure messages is a TensorFlow Dataset or a list/array of strings
        messages_tf = tf.constant(messages.tolist()) # Convert pandas Series to list, then to tf.constant

        tfidf_matrix = self.tfidf_vectorizer(messages_tf) # Use call for transformation

        # Convert SparseTensor to dense numpy array, then to DataFrame
        tfidf_matrix_dense = tfidf_matrix.numpy()

        vocab = self.tfidf_vectorizer.get_vocabulary()

        # Get feature names (vocabulary) after adapting
        feature_names = [f'tfidf_{word}' for word in vocab]

        tfidf_df = pd.DataFrame(
            tfidf_matrix_dense,
            columns=feature_names
        )

        return tfidf_df

    def prepare_features(self, messages, fit_tfidf=False):
        """
        Combine all feature extraction methods with spellchecking
        """

        # Extract base features (before spellcheck)
        base_features = self.extract_base_features(messages)

        # Extract sentiment features (on corrected text)
        sentiment_features = self.extract_sentiment_features(messages)

        # Extract TF-IDF features (on corrected text)
        tfidf_features = self.extract_tfidf_features(messages, fit=fit_tfidf)

        # Combine all features
        all_features = pd.concat([
            base_features.reset_index(drop=True),
            sentiment_features.reset_index(drop=True),
            tfidf_features.reset_index(drop=True)
        ], axis=1)

        all_features = all_features.reindex(columns=self.feature_names, fill_value=0)

        return all_features

    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        predictions = self.model.predict(X_scaled)
        return (predictions >= 0.5).astype(int)

    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return self.model.predict(X_scaled)
    
spam_filter = EnhancedSMSSpamFilter(tfidf_features=3000)
spam_filter.model = ydf.load_model("gubat_model")

spam_filter.scaler = joblib.load('gubat_model_configs/gubat_scaler.joblib')

with open('gubat_model_configs/gubat_tfidf_vocab.json', 'r') as f:
    tfidf_vocab = json.load(f)
tfidf_idf_weights = np.load('gubat_model_configs/gubat_tfidf_idf_weights.npy')

with open('gubat_model_configs/gubat_feature_names.json', 'r') as f:
    feature_names = json.load(f)
spam_filter.feature_names = feature_names

spam_filter.tfidf_vectorizer.set_vocabulary(tfidf_vocab, idf_weights=tfidf_idf_weights)

def predict_new_message(message):
    """
    Predict spam/ham for new messages
    """
    if isinstance(message, str):
        message = [message]

    message_df = pd.DataFrame({'text': message})

    features = spam_filter.prepare_features(message_df['text'], fit_tfidf=False)
    predictions = spam_filter.predict(features)
    probabilities = spam_filter.predict_proba(features)

    results = []
    for i in range(len(message_df)):
        results.append({
            'message': message_df['text'].iloc[i],
            'prediction': str(predictions[i]),
            'probability': float(probabilities[i]),
        })

    return results



app = FastAPI()
security = HTTPBearer()

# Load from environment variables
API_KEY = os.environ['API_KEY']
ENCRYPTION_KEY = os.environ['ENCRYPTION_KEY']

def decrypt_message(encrypted_message: str, key: str) -> str:
    """
    Decrypts an AES-256 message that was encrypted using:
    - GCM Mode
    - A 32-byte (256-bit) key
    - Nonce prepended to the ciphertext (first 12 bytes), followed by ciphertext and MAC
    """
    try:
        encrypted_bytes = base64.b64decode(encrypted_message)

        nonce = encrypted_bytes[:12]
        ciphertext_and_mac = encrypted_bytes[12:]
        key_bytes = key.encode('utf-8')[:32]

        cipher = Cipher(algorithms.AES256(key_bytes), modes.GCM(nonce))
        decryptor = cipher.decryptor()

        ciphertext = ciphertext_and_mac[:-16]
        tag = ciphertext_and_mac[-16:]

        decrypted = decryptor.update(ciphertext) + decryptor.finalize_with_tag(tag)

        return decrypted.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Decryption failed")

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.get("/health")
async def health_check(api_key: str = Depends(verify_api_key)):
    """Health check endpoint - requires valid API key"""
    return {"status": "healthy", "message": "GubatFilterAPI is running"}

class BatchSmsInput(BaseModel):
    messages: List[str]

@app.post("/predict")
async def predict_sms(data: BatchSmsInput,  api_key: str = Depends(verify_api_key)):
    try:
        decrypted_messages = [decrypt_message(msg, ENCRYPTION_KEY) for msg in data.messages]

        results = predict_new_message(decrypted_messages)

        return {
            "predictions": [result['prediction'] for result in results],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid messages")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
