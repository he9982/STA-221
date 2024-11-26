import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from split import load_train
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from lasso_selection import lasso_sel

def cross_validation(cv = 5, clf = None, x = None, y = None,
                     F1 = False, Accuracy = False,
                     Precision = False, Recall = False,
                     use_lasso = False):
    f1_val_scores = []
    f1_train_scores = []
    acc_val_scores = []
    acc_train_scores = []
    pre_val_scores = []
    pre_train_scores = []
    re_val_scores = []
    re_train_scores = []
    

    kf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=True)
    clf = clf

    # Convert y to NumPy array
    # y = np.array(y)

    for train_index, val_index in kf.split(x,y):
        
        # Debug: Ensure no overlap between training and validation indices
        assert set(train_index).isdisjoint(set(val_index)), "Data leakage: Training and validation sets overlap!"
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        #use lasso for feature selection
        if use_lasso:
            try:
                opt_alpha, x_train = lasso_sel(x_train, y_train)
                coef_mask = x_train.shape[1]
                x_val = x_val[:, :coef_mask]
            except Exception as e:
                print(f"Error during Lasso feature selection: {e}")
                continue

        #     alphas = np.arange(0,10,0.5)
        #     for a in alphas:
        #         lasso = Lasso(alpha = alpha,max_iter=10000, 
        #                   random_state=42)
        #         lasso.fit(x_train, y_train)

        #         selection = np.where(lasso.coef_ != 0)[0]
        #         if len(selection) > 0:
        #             x_train = x_train[:, selection]
        #             x_val = x_val[:, selection]
        #         else:
        #             print(f"No features selected with alpha={alphas}. Skipping this fold.")
        #             continue
        # print("OPT Alpha:", a)


        clf.fit(x_train, y_train)

        pred_val = clf.predict(x_val)

        pred_train = clf.predict(x_train)
        
        #metrics
        f1_val = metrics.f1_score(y_val, pred_val, average = 'weighted')
        f1_train = metrics.f1_score(y_train, pred_train, average = 'weighted')
        
        acc_val = metrics.accuracy_score(y_val, pred_val)
        acc_train = metrics.accuracy_score(y_train, pred_train)

        pre_val = metrics.precision_score(y_val, pred_val, average = 'weighted')
        pre_train = metrics.precision_score(y_train, pred_train, average = 'weighted')

        re_val = metrics.recall_score(y_val, pred_val, average = 'weighted')
        re_train = metrics.recall_score(y_train, pred_train, average = 'weighted')
        
        f1_val_scores.append(f1_val)
        f1_train_scores.append(f1_train)

        acc_val_scores.append(acc_val)   
        acc_train_scores.append(acc_train) 

        pre_val_scores.append(pre_val)   
        pre_train_scores.append(pre_train)

        re_val_scores.append(re_val)   
        re_train_scores.append(re_train)
    if F1 == True:
        return f1_val_scores, f1_train_scores
    if Accuracy == True:
        return acc_val_scores, acc_train_scores
    if Precision == True:
        return pre_val_scores, pre_train_scores
    if Recall == True:
        return re_val_scores, re_train_scores
    
def plot(CV = None, train_metrics = None, val_metrics = None):
    Fold = list(range(1, CV + 1))
    train_metric = pd.Series(train_metrics)
    val_metric = pd.Series(val_metrics)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Fold, train_metrics, c = 'b', marker = "^",ls='--',label='Greedy',fillstyle='none')
    ax.plot(Fold, val_metrics, c='g',marker=(8,2,0),ls='--',label='Greedy Heuristic')
    plt.title('Overall Situation of Fitting')
    plt.legend(['Train_F1','Test_F1'])
    plt.show()


def predict(clf = None,
            F1 = None,
            Accuracy = None,
            Recall = None,
            uselasso = False,
            X_train = None, y_train = None, X_test = None, y_test = None):
    # X_train, X_test, y_train, y_test = joblib.load(file_path)
    if uselasso == True:
        try:
            best_alpha, X_train = lasso_sel(X_train, y_train)
            coef_mask = X_train.shape[1]
            X_test = X_test[:, :coef_mask]
        except Exception as e:
            print(f"Error during Lasso feature selection: {e}")
            return None

    clf = clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # print out metrics
    if F1 == True:
        return metrics.f1_score(y_test, pred)
    elif Accuracy == True:
        return metrics.accuracy_score(y_test, pred)
    elif Recall == True:
        return metrics.recall_score(y_test, pred)
    else:
        return metrics.precision_score(y_test, pred)
    



if __name__ == '__main__':
    file_path = '/Users/kuangli/Desktop/UCD/Fall_2024/STA_221/Vision/split_data.pkl'
    X_train, X_test, y_train, y_test = joblib.load(file_path)
    # Debug print
    print(f"X_train type: {type(X_train)}, shape: {np.shape(X_train)}")
    print(f"y_train type: {type(y_train)}, shape: {np.shape(y_train)}")

    f1_val_scores, f1_train_scores = cross_validation(clf = RandomForestClassifier()
, use_lasso=False, 
                     Recall=True, x=X_train, y=y_train)
    print(f"Training F1:", f1_train_scores)
    print(f"Vallidating F1:", f1_val_scores)


    # testf1 = predict(clf = RandomForestClassifier(),F1=True, uselasso=False, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)
    # print(testf1)

    # clf = RandomForestClassifier().fit(X_train, y_train)
    # pred = clf.predict(X_test)
    # print(metrics.f1_score(y_test, pred))




