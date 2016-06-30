# This file contains a framework for the IBM contest, including:
# - ingesting the training and evaluation data sets
# - creating a model from the training set
# - applying the model to the test set and creating a submission file

import sys
import argparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

#classifier names and the sklearn classifiers
# You will be adding to these variables throughout the class
names = ['KNN', 'DecisionTree']
classifiers = [KNeighborsClassifier(1), DecisionTreeClassifier()]


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


#Can change if you like, but not required
def train_and_label(train_filename, test_filename, clf, pickled):
    """ Simple cycle of creating a classifier from the training data and
    applying it to the test data, returning the example IDs which are true for contest submission
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
    results = []
    for i in range(len(labels)):
        if labels[i] == 'true':
            results.append(test_data['id'][i])
    return results


#This one you will modify
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
    parser.add_argument("--clf", help="Optional name of classifier, defaults to KNN", type=str, default="KNN")
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

    answers = train_and_single_label(args.train_file, args.test_file, model, pickled)
    for a in answers: print a