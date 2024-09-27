import numpy as np
import pandas as pd
import re
import os
import joblib
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

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

lemma = WordNetLemmatizer()
swords = stopwords.words("english")

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data.columns = data.columns.str.strip()
    # print(data.columns)
    # print(data['Title'].head())
    return data

def process_text(text):
    if pd.isna(text):
        return " "
    text = re.sub(r'http\S+', '', text)
    text = re.sub("[^a-zA-Z0-9]"," ",text)
    text = nltk.word_tokenize(text.lower())
    text = [lemma.lemmatize(tex) for tex in text]
        # text = [tex for tex in text if text not in swords]
    text = [tex for tex in text if len(tex) > 2 and  tex not in swords]
    cleaned_text = " ".join(text)
    return cleaned_text




    


def main():
    data = load_data('Desktop/UCD/Fall 2024/STA 221/final.csv')
    data['cleaned'] = data['Full Text'].apply(process_text)

    le = LabelEncoder()
    data['Sentiment'] = le.fit_transform(data['Sentiment']) 

    output_file = 'Desktop/UCD/Fall 2024/STA 221/cleaned.csv'
    data[['cleaned', 'Sentiment']].to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

