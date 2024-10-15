from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer  # Importing TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib  # To save the model and data splits
from process import load_data  # Assuming load_data() is defined in process.py
from split import split_data, load_train, load_test
from split import load_train
import numpy as np



def train_model(model = None, vect = None, LASSO = False):
    X_train, y_train = load_train()
    X_train = np.array([str(x) if x is not None else "" for x in X_train])
    vectorize = vect
    vect_x_train = vectorize.fit_transform(X_train)
    print("Vectorization Shape: ", vect_x_train.shape)

    if LASSO == True:
        l1 = SelectFromModel(estimator=LogisticRegression(penalty = 'l1', solver = 'liblinear', C = 1,
                                                          random_state=0)).fit(vect_x_train,y_train)
        l1_support = l1.get_support()
        l1_feature = vect_x_train[:, l1_support]
        vect_x_train = l1_feature
        joblib.dump(l1, 'lasso_feature_selection.pkl')
        # print(f"Number of feature selected: {l1_support.sum()} out of {vect_x_train.shape[1]}")
        # print("Lasso Feature Shape: ", vect_x_train.shape)


    print("TFIDF Vectorization complete")
    print("Vectorization Shape: ", vect_x_train.shape)
    clf = model
    clf.fit(vect_x_train, y_train) #fit model
    print("model complete")
    print('Training Complete')
    joblib.dump(model, 'trained_model.pkl') #save model
    joblib.dump(vectorize, 'vectorization.pkl')
    print("Model saved as 'trained_model.pkl'. Vectorozation saved")
    return clf

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data('Desktop/UCD/Fall_2024/STA_221/cleaned.csv')
    train_model(MultinomialNB(), vect = TfidfVectorizer(), LASSO=False)
    