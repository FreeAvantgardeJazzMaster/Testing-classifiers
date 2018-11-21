import numpy as np
import matplotlib.pyplot as plt
from utils.load import convert_to_onehot
from utils.evaluate import scorer_squared_error, scorer_01loss
from utils.load import read_and_convert_pandas_files
import time

def evaluate_accuracy_and_time(classifier, X_train, y_train, X_test, y_test):
    data_to_plot = {}

    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time
    data_to_plot.update({"training_time": training_time})
    print("Training time = {0}".format(training_time))

    scorers = [(scorer_01loss, "0/1 loss"), (scorer_squared_error, "squared error")]
    start_time = time.time()
    for scorer, scorer_name in scorers:
        print("Train {0} = {1}".format(scorer_name, scorer(classifier, X_train, y_train)))
        data_to_plot.update({"train_squared_error": scorer(classifier, X_train, y_train)})
        print("Test {0} = {1}".format(scorer_name, scorer(classifier, X_test, y_test)))
        data_to_plot.update({"test_squared_error": scorer(classifier, X_test, y_test)})
    testing_time = time.time() - start_time
    data_to_plot.update({"testing_time": testing_time})
    print("Testing time = {0}".format(testing_time))
    return data_to_plot