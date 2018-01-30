"""
Contains all the necessary functions to evaluate trained models and generate
validation, learning, iteration and timing curves.

"""
from helpers import load_pickled_model, get_abspath
from model_train import split_data, balanced_accuracy
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit
import pandas as pd
import numpy as np
import timeit
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn


def basic_results(grid, X_test, y_test, data_name, clf_name):
    """Gets best fit against test data for best estimator from a particular
    grid object. Note: test score funtion is the same scoring function used
    for training.

    Args:
        grid (GridSearchCV object): trained grid search object.
        X_test (numpy.Array): Test features.
        y_test (numpy.Array): Test labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.

    """
    # set columns and data points to be extracted from grid object
    cols = ['best_estimator', 'best_score', 'best_params', 'test_score']
    data = [[grid.best_estimator_, grid.best_score_,
             grid.best_params_, grid.score(X_test, y_test)]]

    # write results to a data frame
    parentdir = 'results'
    results = pd.DataFrame(columns=cols, data=data)
    target = '{}/{}'.format(parentdir, clf_name)
    resfile = get_abspath('{}_basic_results.csv'.format(data_name), target)
    results.to_csv(resfile, index=False)


def create_learning_curve(estimator, scorer, X_train, y_train, data_name, clf_name, cv=5):
    """Generates a learning curve for the specified estimator, saves tabular
    results to CSV and saves a plot of the learning curve.

    Args:
        estimator (object): Target classifier.
        scorer (object): Scoring function to be used.
        X_train (numpy.Array): Training features.
        y_train (numpy.Array): Training labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.
        cv (int): Number of folds in cross-validation splitting strategy.

    """
    # set training sizes and intervals
    train_sizes = np.arange(0.01, 1.0, 0.01)

    # set cross validation strategy to use StratifiedShuffleSplit
    cv_strategy = StratifiedShuffleSplit(n_splits=cv, random_state=0)

    # create learning curve object
    LC = learning_curve(estimator, X_train, y_train, cv=cv_strategy,
                        train_sizes=train_sizes, scoring=scorer, n_jobs=6)

    # extract training and test scores as data frames
    train_scores = pd.DataFrame(index=LC[0], data=LC[1])
    test_scores = pd.DataFrame(index=LC[0], data=LC[2])

    # save data frames to CSV
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    train_file = get_abspath('{}_LC_train.csv'.format(data_name), res_tgt)
    test_file = get_abspath('{}_LC_test.csv'.format(data_name), res_tgt)
    train_scores.to_csv(train_file, index=False)
    test_scores.to_csv(test_file, index=False)

    # create learning curve plot
    plt.figure(1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1),
             marker='.', color='b', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1),
             marker='.', color='g', label='Cross-validation score')
    plt.legend()
    plt.grid(linestyle='dotted')
    plt.xlabel('Samples used for training as a percentage of total')
    plt.ylabel('Accuracy')

    # save learning curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_LC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_timing_curve(estimator, dataset, data_name, clf_name):
    """Generates a timing curve for the specified estimator, saves tabular
    results to CSV and saves a plot of the timing curve.

    Args:
        estimator (object): Target classifier.
        dataset(pandas.DataFrame): Source data set.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.

    """
    # set training sizes and intervals
    train_sizes = np.arange(0.01, 1.0, 0.01)

    # initialise variables
    train_time = []
    predict_time = []
    df_final = []

    # iterate through training sizes and capture training and predict times
    for i, train_data in enumerate(train_sizes):
        X_train, X_test, y_train, y_test = split_data(
            dataset, test_size=1 - train_data)
        start_train = timeit.default_timer()
        estimator.fit(X_train, y_train)
        end_train = timeit.default_timer()
        estimator.predict(X_test)
        end_predict = timeit.default_timer()
        train_time.append(end_train - start_train)
        predict_time.append(end_predict - end_train)
        df_final.append([train_data, train_time[i], predict_time[i]])

    # save timing results to CSV
    timedata = pd.DataFrame(data=df_final, columns=[
                            'Training Data Percentage', 'Train Time', 'Test Time'])
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    timefile = get_abspath('{}_timing_curve.csv'.format(data_name), res_tgt)
    timedata.to_csv(timefile, index=False)

    # generate timing curve plot
    plt.figure(2)
    plt.plot(train_sizes, train_time, marker='.', color='b', label='Train')
    plt.plot(train_sizes, predict_time, marker='.', color='g', label='Predict')
    plt.legend()
    plt.grid(linestyle='dotted')
    plt.xlabel('Samples used for training as a percentage of total')
    plt.ylabel('Elapsed user time in seconds')

    # save timing curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_TC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


if __name__ == '__main__':
    # set seed for cross-validation sampling
    seed = 0

    # set scoring function
    scorer = make_scorer(balanced_accuracy)

    # load datasets
    p_wine = get_abspath('winequality.csv', 'data/experiments')
    p_seismic = get_abspath('seismic-bumps.csv', 'data/experiments')
    df_wine = pd.read_csv(p_wine)
    df_seismic = pd.read_csv(p_seismic)
    dfs = {'wine': df_wine, 'seismic': df_seismic}
    dnames = ['wine', 'seismic']

    # instantiate dict of estimators
    estimators = {'KNN': None,
                  'DT': None,
                  'ANN': None,
                  'SVM': None,
                  'Boosting': None}
    mnames = ['KNN', 'DT', 'ANN', 'SVM', 'Boosting']

    # start model evaluation loop
    for df in dnames:
        X_train, X_test, y_train, y_test = split_data(dfs[df], seed=seed)
        # load pickled models into estimators dict
        for m in mnames:
            mfile = '{}/{}_best_estimator.pkl'.format(m, df)
            print "Estimator: " + mfile + " is being used for " + df + " data"
            model = load_pickled_model(get_abspath(mfile, filepath='models'))
            estimators[m] = model

        # generate validation, learning, timing and iteration curves/plots
        for name, estimator in estimators.iteritems():
            # basic_results(grid, X_test, y_test, data_name=df, clf_name=name)
            create_learning_curve(estimator, scorer, X_train,
                                  y_train, data_name=df, clf_name=name)
            create_timing_curve(estimator, dataset=dfs[
                                df], data_name=df, clf_name=name)
