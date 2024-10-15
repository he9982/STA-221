import joblib 
from sklearn.metrics import accuracy_score
from split import load_test


def predict(LASSO = False):
    X_test, y_test = load_test()
    tfidf = joblib.load('vectorization.pkl')
    X_test = tfidf.transform(X_test)
    if LASSO == True:
        lasso = joblib.load('lasso_feature_selection.pkl')
        X_test = lasso.transform(X_test)
    clf = joblib.load('trained_model.pkl')
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    predict(LASSO = False)