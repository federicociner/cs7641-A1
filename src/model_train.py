"""
Contains all the necessary functions to find the best parameters for each
algorithm and train that estimator.

"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from helpers import get_datafile, save_model
from clf_decision_tree import DecisionTree
import clf_knn as KNN
import clf_neural_network as MLPClassifier
import clf_svm as SVM
import clf_adaboost as AdaBoost
import pandas as pd
import numpy as np


def balanced_accuracy(labels, predictions):
    """Modifies the standard accuracy score to account for potential
    imbalances in class distributions.

    Args:
        labels (numpy.array): actual class labels.
        predictions (numpy.array): predicted class labels.

    """
    weights = compute_sample_weight('balanced', labels)
    return accuracy_score(labels, predictions, sample_weight=weights)


def split_data(df, test_size=0.3, seed=0):
    """Prepares a data frame for model training and testing by converting data
    to Numpy arrays and splitting into train and test sets.

    """
    # convert data frame to Numpy array and split X and y
    X_data = df.drop(columns='class').as_matrix()
    y_data = df['class'].as_matrix()

    # split into train and test sets, ensuring that composition of classes in
    # original dataset is maintained in the splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=seed, stratify=y_data)

    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, clf, scorer, n_jobs=6, cv=5, seed=0):
    """Trains model using GridSearchCV and returns object containing results.

    Args:
        X_train (numpy.Array): training features.
        y_train (numpy.Array): training labels.
        clf (object): classifier object with pipeline and params attributes.
        scorer (object): scoring function.
        n_jobs (int): number of jobs to run in parallel.
        cv (int): determine cross-validation splitting strategy.
    Returns:
        grid (GridSearchCV object):

    """
    # get pipeline and params from classifier object
    pipeline = clf.ipeline
    params = clf.params

    # generate grid search object
    grid = GridSearchCV(estimator=pipeline, n_jobs=6, param_grid=params,
                        cv=cv, scoring=scorer, refit=True)

    # initiate grid search
    grid.fit(X_train, y_train)

    return grid


def run(filename, filepath):

if __name__ == '__main__':
    # set seed for cross-validation sampling
    seed = 0

    # set balance_accuracy scoring functino
    scorer = make_scorer(balanced_accuracy)
