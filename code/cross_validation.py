import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from process import load_data
from split import load_train
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cross_validation(cv = 5, clf = None, vec = None, x = None, y = None, F1 = False, Accuracy = False,
                     Precision = False, Recall = False):
    f1_val_scores = []
    f1_train_scores = []
    acc_val_scores = []
    acc_train_scores = []
    pre_val_scores = []
    pre_train_scores = []
    re_val_scores = []
    re_train_scores = []

    kf = StratifiedKFold(n_splits=cv, random_state=0, shuffle=True)
    vect = vec
    vect_x = vect.fit_transform(x)
    l1 = SelectFromModel(estimator=LogisticRegression(penalty = 'l1', solver = 'liblinear')).fit(vect_x,y)
    l1_support = l1.get_support()
    l1_feature = vect_x[:,l1_support]
    l1_feature.shape
    clf = clf

    # Convert y to NumPy array
    y = np.array(y)

    for train_index, val_index in kf.split(l1_feature,y):
        x_train, x_val = l1_feature[train_index], l1_feature[val_index]
        y_train, y_val = y[train_index], y[val_index]

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



def main():
    X_train, y_train = load_train()
    print(f'Loaded train data size: {len(X_train)}')
    f1_val_scores, f1_train_scores = cross_validation(cv = 5, clf = MultinomialNB(), vec = TfidfVectorizer(), x = X_train, y = y_train, F1 = True)
    print(f1_val_scores)
    print(f1_train_scores)
    plot(CV = 5, train_metrics = f1_train_scores, val_metrics = f1_val_scores)



if __name__ == '__main__':
    main()



        