'''
Author: Kalina Jasinska
'''

from plot_learning_curve import evaluate_accuracy_and_time
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from utils.evaluate import scorer_squared_error, scorer_01loss
from utils.load import read_datasets
import matplotlib.pyplot as plt
import time
import numpy as np
# Implement plotting of a learning curve using sklearn
# Remember that the evaluation metrics to plot are 0/1 loss and squared error


datasets = [('../data/badges2-train.csv', '../data/badges2-test.csv',  "Badges2"),
            ('../data/credit-a-train.csv','../data/credit-a-test.csv', "Credit-a"),
            ('../data/credit-a-mod-train.csv','../data/credit-a-mod-test.csv', "Credit-a-mod"),
            ('../data/spambase-train.csv', '../data/spambase-test.csv', "Spambase"),
            ('../data/covtype-train.csv', '../data/covtype-test.csv', "Covtype")
           ]

data_to_plot = defaultdict(list)

def make_learning_curves():
    global data_to_plot
    for key, value in data_to_plot.items():
        plt.title('Charts')
        plt.plot(range(10, len(value)*10 + 1, 10), value, label=key)
    plt.legend()
    plt.show()


def evaluate_classifer():
    global data_to_plot

    fn, fn_test, ds_name = '../data/spambase-train.csv', '../data/spambase-test.csv', "Spambase"
    print("Dataset {0}".format(ds_name))
    for prc in range(1, 11):
        prc *= 0.1
        X_train, y_train, X_test, y_test, is_categorical = read_datasets(fn, fn_test, prc)
        classifier = LogisticRegression()
        results = evaluate_accuracy_and_time(classifier, X_train, y_train, X_test, y_test)
        for key, value in results.items():
            data_to_plot[key].append(value)

    print(data_to_plot)
if __name__ == "__main__":
    evaluate_classifer()
    make_learning_curves()