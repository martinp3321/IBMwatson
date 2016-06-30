# This file runs cross-validation and stratified cross-validationon a subset of the IBM data

import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn import cross_validation
#handy linear algebra and related library
import numpy as np


#Don't worry about understanding all the details of this code
def k_fold_eval(traind, k=5, clf=DecisionTreeClassifier(max_depth=10)):
    '''
    Perform k-fold cross validation on the traind,
    print precision and std deviation and recall and std deviation
    Inputs:
    traind: dictionary in format returned by contest.extract_ibm_data
    k: number of folds
    clf: classifier to use to train the data in each fold
    '''
    kf = cross_validation.KFold(len(traind['data']), n_folds=k)
    X = np.array(traind['data'])
    y = np.array(traind['target'])
    results = run_folds(kf, clf, X, y)
    print "K-fold XVal Precision and recall for classifier:", clf.__class__
    print "P: %0.2f (+/- %0.2f); R: %0.2f (+/- %0.2f)" \
          % (results[0]/k, results[2], results[1]/k, results[3])


#Don't worry about understanding all the details of this code
def stratified_k_fold_eval(traind, k=5, clf=DecisionTreeClassifier(max_depth=10)):
    '''
    Perform stratified k-fold cross validation on the traind,
    print precision and std deviation and recall and std deviation
    Inputs:
    traind: dictionary in format returned by contest.extract_ibm_data
    k: number of folds
    clf: classifier to use to train the data in each fold
    '''

    X = np.array(traind['data'])
    y = np.array(traind['target'])
    skf = cross_validation.StratifiedKFold(y, n_folds=5)
    results = run_folds(skf, clf, X, y)
    print "Stratified K-fold XVal Precision and recall for classifier:", clf.__class__
    print "P: %0.2f (+/- %0.2f); R: %0.2f (+/- %0.2f)" \
          % (results[0]/k, results[2], results[1]/k, results[3])


def compute_pri(actual, predicted):
    '''
    Return a tuple of the precision, recall, and ibm score of the predicted labels
    wrto the actual labels
    Input: lists of labels
    Output: (p, r, i), your calculated precision, recall, and ibm score values
    '''
    precision = 0
    recall = 0
    ibm_score = 0
    '''*** YOUR CODE HERE ***'''
    return precision, recall, ibm_score


#Don't worry about understanding all the details of this code
def run_folds(folds, clf, X, y):
    ''' Support method for KFold & stratified KFold
    '''
    tot_precision = 0.0
    tot_recall = 0.0
    tot_ibm = 0.0
    p_list = []
    rec_list=[]
    for train, test in folds:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        labels = clf.predict(X_test)

        #call the new method that you write
        pr, rec, i = compute_pri(y_test, labels)

        tot_precision += pr
        p_list.append(pr)
        tot_recall += rec
        rec_list.append(rec)
        tot_ibm += i
    p_list = np.array(p_list)
    rec_list = np.array(rec_list)
    print "Average IBM score:", tot_ibm/len(folds)
    return tot_precision, tot_recall, np.std(p_list), np.std(rec_list)


#main method for running cross validation, prints P/R and (IBM score)/N to terminal
if __name__ == '__main__':

    subset = joblib.load('subset_trn.pkl')
    k_fold_eval(subset)
    stratified_k_fold_eval(subset)


