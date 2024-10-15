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
        coef_mask = lasso.coef_ != 0
        x_sel = X[:, coef_mask]
        
        if x_sel.shape[1] == 0:
            continue
        try:
            score = cross_val_score(LogisticRegression(), x_sel, y, cv = 5, scoring = 'accuracy')
            avg_acc = np.mean(score)

            if avg_acc > best_score:
                best_score = avg_acc
                opt_alpha = a
                print("Optimal Alpha:", a)
                best_lassoAlpha = Lasso(alpha = a)
                best_lassoAlpha.fit(X, y)
                final_x_sel = x_sel[:, best_lassoAlpha.coef_ != 0]
        except ValueError as e:
            print(f"Encounter value error with alpha {a}: {e}")
    print('Optimal Alpha:', opt_alpha)
    return final_x_sel if 'final_x_sel' in locals() else None

if __name__ == "__main__":
    X_train, y_train = load_train()
    tfidf_vec = joblib.load('tfidf_vectorization.pkl')
    X_train = tfidf_vec.fit_transform(X_train)
    sel_fet = lasso_sel_alpha(X_train, y_train)
    if sel_fet is not None:
        print("Successful feature selection from lasso", sel_fet)
    else:
        print("Failed to select feature with lasso")
