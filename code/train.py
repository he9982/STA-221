from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Or any other model of your choice
from sklearn.metrics import accuracy_score
import joblib  # To save the model and data splits
from process import load_data  # Assuming load_data() is defined in process.py
from split import split_data, load_train, load_test
from split import load_train
import numpy as np



def train_model(model = None):
    X_train, y_train = load_train()
    X_train = np.array([str(x) if x is not None else "" for x in X_train])
    tfidfvec = TfidfVectorizer(max_features=100000) 
    X_train = tfidfvec.fit_transform(X_train)

    print("TFIDF Vectorization complete")

    clf = model
    clf.fit(X_train, y_train) #fit model
    print("model complete")
    print('Training Complete')
    joblib.dump(model, 'trained_model.pkl') #save model
    joblib.dump(tfidfvec, 'tfidf_vectorization.pkl')
    print("Model saved as 'trained_model.pkl'. Vectorozation saved")
    return clf

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data('Desktop/UCD/Fall 2024/STA 221/cleaned.csv')
    train_model(RandomForestClassifier())
    