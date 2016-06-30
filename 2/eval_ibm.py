# This file runs stratified cross-validation and LOOCV on a subset of the IBM data

import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import random
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support
#handy linear algebra and related library
import numpy as np


#this is just FYI, how I created the subset of the training data stored in subset_train.pkl
def shuffle_and_split(traind):
    '''
    Takes a random 10% sample of the IBM data and returns a tuple of X, y and q_id
    '''
    zipd = zip(traind['data'], traind['target'], traind['q_id'])
    #randomize the order; you could also save this in a serialized file to experiment with later, with:
    # joblib.dump(zipd, 'zipShuffleTrn.pkl')
    random.shuffle(zipd)

    N = int(len(traind['data']) * 0.1)
    ten_perc_z = zipd[:N]
    X = [l[0] for l in ten_perc_z]
    y = [l[1] for l in ten_perc_z]
    q_ids = [l[2] for l in ten_perc_z]
    return {'data': X, 'target': y, 'q_id': q_ids}


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


#Don't worry about understanding all the details of this code
def run_folds(folds, clf, X, y):
    ''' Support method for KFold & stratified KFold
    '''
    tot_precision = 0.0
    tot_recall = 0.0
    p_list = []
    rec_list=[]
    for train, test in folds:
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        labels = clf.predict(X_test)
        #print "labels: ",labels
        #print "correct:",y_test
        pr,rec,_,_ = precision_recall_fscore_support(y_test, labels, pos_label='true')
        #print "p: %f, r: %f" % (pr[1], rec[1])
        tot_precision += pr[1]
        p_list.append(pr[1])
        tot_recall += rec[1]
        rec_list.append(rec[1])
    p_list = np.array(p_list)
    rec_list = np.array(rec_list)
    return tot_precision, tot_recall, np.std(p_list), np.std(rec_list)


#main method for running cross validation, prints P/R and (IBM score)/N to terminal
if __name__ == '__main__':

    subset = joblib.load('subset_trn.pkl')
    k_fold_eval(subset)
    stratified_k_fold_eval(subset)


