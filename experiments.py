# experiments filtering noisy examples
from sklearn.externals import joblib
import argparse
import math
import numpy as np
import random
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def scale(train, test):
    """ Scales the data so all features are uniformly scaled.
    Just FYI, I have given you the data in this version
    (You need to do it for both training and testing set for it to work)
    """
    scaler = preprocessing.StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)


def train_and_single_label(train_data, test_data, clf):
    """ Only return one example ID for each q_id
    """
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


#for experiments duplicating the True examples
def weigh_exs(datad, replicate=5):
    """replicate the true examples in the dictionary.
    WARNING: changes original dictionary
    """
    x = datad['data']
    newx = []
    new_ids = []
    new_qids = []
    new_targs = []
    labels = datad['target']
    for i in range(len(x)):
        if labels[i] == 'true':
            newx.extend([x[i]]*replicate)
            new_ids.extend([datad['id'][i]]*replicate)
            new_qids.extend([datad['q_id'][i]]*replicate)
            new_targs.extend(['true']*replicate)
    print "%d trues" % (len(new_targs)/replicate)
    labels.extend(new_targs)
    datad['id'].extend(new_ids)
    datad['q_id'].extend(new_qids)
    return {'data': np.concatenate((x, newx)), 'target': labels, 'id': datad['id'],
            'q_id': datad['q_id']}


if __name__ == '__main__':
    #parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="Name of file with training data", type=str)
    parser.add_argument("test_file", help="Name of file with evaluation data", type=str)
    parser.add_argument("--clf", help="Optional name of classifier, defaults to Logistic", type=str, default="Logistic")
    args = parser.parse_args()

    #load original pickled file first
    traind = joblib.load('train-red.pkl')
    normd_trn = joblib.load('justScaledX-trn-red.pkl')
    traind['data'] = normd_trn

    testd = joblib.load('Scaled_reduced_test.pkl')

    teamOne = True
    if teamOne:
        answers = train_and_single_label(traind, testd, RandomForestClassifier(n_estimators=500))
    else:
        weighted_traind = weigh_exs(traind)
        answers = train_and_single_label(weighted_traind, testd, RandomForestClassifier(n_estimators=500))
    for a in answers: print a