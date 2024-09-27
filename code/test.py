import joblib 
from sklearn.metrics import accuracy_score
from split import load_test


def predict():
    X_test, y_test = load_test()
    tfidf = joblib.load('tfidf_vectorization.pkl')
    X_test = tfidf.transform(X_test)
    clf = joblib.load('trained_model.pkl')
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    predict()