import numpy as np
import pandas as pd
import re
import os
import joblib
import nltk
import ssl
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize


# Ensure all necessary NLTK resources are downloaded
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

lemma = WordNetLemmatizer()
swords = stopwords.words("english")

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data.columns = data.columns.str.strip()
    # print(data.columns)
    # print(data['Title'].head())
    return data

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if the tag is not in the dictionary


# Tokenization and lemmatization function with POS tagging
def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    lemmatized_tokens = []
    for token in tokens:
        pos = get_wordnet_pos(token)
        lemmatized_token = lemma.lemmatize(token, pos)
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens



def process_text(text):
    if pd.isna(text):
        return " "
    text = re.sub(r'http\S+', '', text)
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    # text = nltk.word_tokenize(text.lower())
    tokens = tokenize_and_lemmatize(text)
    tokens = [token for token in tokens if len(token) > 2 and token not in swords]
    cleaned_text = " ".join(tokens)
    return cleaned_text



    


def main():
    file_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/BA_Romdom.csv'
    data = load_data(file_path)
    data['cleaned'] = data['Full Text'].apply(process_text)

    # Encode Sentiment: Map pos, neg, neu to numerical values
    label_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    data['Sentiment'] = data['Sentiment'].map(label_mapping)

    #check distribution
    sentiment_percentage = data['Sentiment'].value_counts(normalize=True) * 100
    print("Sentiment Percentage: ", sentiment_percentage)

    output_file = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/cleaned.csv'
    data[['cleaned', 'Sentiment']].to_csv(output_file, index=False)
    # print(data['cleaned'])


if __name__ == '__main__':
    main()

