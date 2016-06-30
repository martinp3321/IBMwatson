# Running this file will create a pickled version of the IBM data dictionaries
# for the training and test sets to save you time running different models on the data.
# if you change the data structure of the dictionary, you will need to rerun this file

import argparse
import contest
from sklearn.externals import joblib

### Usage: python contest.py train_file test_file
if __name__ == '__main__':
    #parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file", help="Name of file with training data", type=str)
    parser.add_argument("test_file", help="Name of file with evaluation data", type=str)
    args = parser.parse_args()

    train_dictionary = contest.extract_ibm_data(args.train_file)
    #If you want to name it something else, change the line below
    joblib.dump(train_dictionary, "train.pkl")

    test_dictionary = contest.extract_ibm_data(args.test_file, test_file=True)
    #Again, you can change the name below if you like
    joblib.dump(test_dictionary, "test.pkl")