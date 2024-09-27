from process import load_data
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def split_data(path):
    df = pd.read_csv(path)
    X,y = df['cleaned'], df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #20% test, 80% train
    X_train = [str(x) if x is not None else "" for x in X_train]
    X_test = [str(x) if x is not None else "" for x in X_test]
    joblib.dump((X_train, X_test, y_train, y_test), 'split_data.pkl')

    return X_train, X_test, y_train, y_test

def load_train():
    X_train, _, y_train, _ = joblib.load('split_data.pkl')
    return X_train, y_train

def load_test():
    _, X_test, _, y_test = joblib.load('split_data.pkl')
    return X_test, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data('Desktop/UCD/Fall 2024/STA 221/cleaned.csv')
    print('Shape of train', len(X_train))
    print('Shape of test', len(X_test))
