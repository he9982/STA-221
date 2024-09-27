import joblib
import pandas as pd
import numpy as np
from split import load_train, load_test
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression

def lasso_sel_alpha(X,y):
    best_score = 0
    opt_alpha = None
    alphas = np.arange(0,10,0.5)
    
    for a in alphas:
        lasso = Lasso(alpha = a)
        lasso.fit(X, y)
        x_sel = X[:, lasso.coef_ != 0]
        score = cross_val_score(LogisticRegression(), x_sel, y, cv = 5, scoring = 'accuracy')
        avg_acc = np.mean(score)

        if avg_acc > best_score:
            best_score = avg_acc
            opt_alpha = a
            print("Optimal Alpha:", a)
            best_lassoAlpha = Lasso(alpha = a)
            best_lassoAlpha.fit(X, y)
            final_x_sel = x_sel[:, best_lassoAlpha.coef_ != 0]
    print('Best Alpha:', opt_alpha)
    return final_x_sel

if __name__ == "__main__":
    X_train, y_train = load_train()
    tfidf_vec = joblib.load('tfidf_vectorization.pkl')
    X_train = tfidf_vec.fit_transform(X_train)
    lasso_sel_alpha(X_train, y_train)
