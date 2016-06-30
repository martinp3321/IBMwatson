# - ingesting the training and evaluation data sets
# - creating a model from the training set
# - applying the model to the test set and creating a submission file

import sys
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
import numpy as np
import math

#classifier names and the sklearn classifiers
# You will be adding to these variables throughout the class
# We haven't talked about all of these, but you might want to look at them
names = ['KNN', 'DecisionTree', 'Logistic', 'NB', 'RF', 'Boost']
classifiers = [KNeighborsClassifier(1), DecisionTreeClassifier(), LogisticRegression(), 
               GaussianNB(),
               RandomForestClassifier(n_estimators=100), AdaBoostClassifier()]


#You shouldn't need to change this
def extract_ibm_data(filename, test_file=False):
    """ extract data from the IBM contest training or evaluation files
    Inputs: Name of the file
    test_file: if True, we will not look for the target value, since it is not included

    Returns: a dictionary with keys 'target_names', 'target', 'q_id' and 'data'
    """
    try:
        in_file = open(filename, 'rU')
        s = in_file.read()
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        raise
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    #split by line
    rows = s.strip().split('\n')
    #initialization
    datadict = {'target_names': ['true', 'false'], 'target': [], 'data': [], 'q_id': [], 'id': []}
    for row in rows:
        cols = row.split(',')
        if not test_file:
            #target classes are in last column
            datadict['target'].append(cols[-1])
            #take out target value
            cols = cols[:-1]
        #example ID in first column
        datadict['id'].append(cols[0])
        #question ID is in second column; reading in makes it a float but we want ints
        datadict['q_id'].append(int(float(cols[1])))
        #now we don't need question ID; also remove the first col, which is just an example ID
        cols = cols[2:]
        row_data = [cvrt_to_num_if_can(c) for c in cols]
        datadict['data'].append(row_data)

    return datadict


def proportion_true(y):
    ''' returns the proportion of true labels in the output, y
    '''
    return float( len([i for i in y if i=='true']))/len(y)

#Can change if you like, but not required
def train_and_label_proba(train_filename, test_filename, clf, pickled):
    """ This adds to the original train_and_label by also looking at the
    "confidence" of the classifier's output, in the form of "predict_proba"
    and only picking the true's that are high.
    Orders the classifier's confidence outputs and stops when it finds its 
    first high confidence false
    Still returns the example IDs that are true for use in contest submission
    """

    if pickled:
        train_data = joblib.load(train_filename)
        test_data = joblib.load(test_filename)
    else:
        train_data = extract_ibm_data(train_filename)
        test_data = extract_ibm_data(test_filename, test_file=True)

    X = train_data['data']
    y = train_data['target']
    #print "training"
    clf.fit(X, y)

    trues_in_train = proportion_true(y)
    n_trues_in_test = int(len(test_data['data']) * trues_in_train)
    #print "%d training exs, %f prop true. %d test, %d should be true" %(len(y), trues_in_train, len(test_data['data']), n_trues_in_test)
    labels = clf.predict(test_data['data'])
    print "%d trues originally" %len([i for i in labels if i=='true'])

    #get index of true in the classes
    t_pos = np.where(clf.classes_ == 'true')[0][0]
    #get the probabilities associated with true
    probas = [x[t_pos] for x in clf.predict_proba(test_data['data'])]

    #print "getting results"
    #get reversed sorted indexes into probas (argsort goes lowest to highest)
    sorted_proba_idx = np.argsort(probas)[::-1]
    n_trues_found = 0
    results = []
    #stop gathering true labels in two situations, 1 and 2 below
    for i in sorted_proba_idx:
        #1. stop if we've found more trues than we probably should
        if n_trues_found > n_trues_in_test:
            break
        if labels[i] == 'true':
            results.append(test_data['id'][i])
            n_trues_found += 1
        #2. stop if we hit a false label (rest of trues are less probable)
        else:
            break

    return results


#Solution for HW1
def train_and_single_label(train_filename, test_filename, clf, pickled):
    """ Only return one example ID for each q_id
    """
    if pickled:
        train_data = joblib.load(train_filename)
        test_data = joblib.load(test_filename)
    else:
        train_data = extract_ibm_data(train_filename)
        test_data = extract_ibm_data(test_filename, test_file=True)

    X = train_data['data']
    y = train_data['target']
    clf.fit(X, y)

    labels = clf.predict(test_data['data'])
    #now manipulate the results using test_data['q_id'] to filter the labels
    ##NEW CODE:
    used_qids = []
    results = []
    for i in range(len(labels)):
        if labels[i] == 'true':
            if not test_data['q_id'][i] in used_qids:
                results.append(test_data['id'][i])
                used_qids.append(test_data['q_id'][i])
    return results


#Utility methods from here until main; You shouldn't need to modify this code

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def is_int(str):
    try:
        int(str)
        return True
    except ValueError:
        return False


def any_float(in_list):
    '''
    Are any of the elements of in_list floats?
    '''
    for i in in_list:
        if is_int(i):
            continue
        if is_float(i):
            return True
    return False


def cvrt_to_num_if_can(str, prefer_float=False):
    ''' If str is really a number,
    convert it to same, preferring floats if flag is True,
    else prefering ints
    '''

    if prefer_float and is_float(str):
        return float(str)
    if is_int(str):
        return int(str)
    if is_float(str):
        return float(str)
    return str


def cnvrt_list_to_nums(in_list):
    return [cvrt_to_num_if_can(x) for x in in_list]


def check_classifier(clf_name):
    for name, clf in zip(names, classifiers):
        if name == clf_name:
            return clf



#main method for creating a contest entry; dumps output to terminal, appropriate for piping to a file
### Usage: python contest.py train_file test_file {--clf classifier_name} {--csv}
if __name__ == '__main__':
    #parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="Name of file with training data", type=str)
    parser.add_argument("test_file", help="Name of file with evaluation data", type=str)
    parser.add_argument("--noProb", help="Pick first instead of highest prob", action="store_true")
    parser.add_argument("--clf", help="Optional name of classifier, defaults to Logistic", type=str, default="Logistic")
    parser.add_argument("--csv", help="If provided, reads the original IBM csv files, otherwise reads joblib dump files",
                        action="store_true")
    args = parser.parse_args()

    model = check_classifier(args.clf)
    if model is None:
        print "Invalid classifier name %s. Did you forget to add it to the program at the top?" %args.clf
        print "Exiting program"
        sys.exit()

    pickled = True
    if args.csv:
        pickled = False

    if args.noProb:
        answers = train_and_single_label(args.train_file, args.test_file, model, pickled)
    else:
        answers = train_and_label_proba(args.train_file, args.test_file, model, pickled)
    for a in answers: print a