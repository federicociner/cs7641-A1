from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


class SVM(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline([('Scale', StandardScaler()), ('SVM', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVM__kernel': ['linear', 'poly', 'rbf'],
                       'SVM__C': 10.0 ** np.arange(-3, 8),
                       'SVM__gamma': 10.0 ** np.arange(-5, 4),
                       'SVM__cache_size': [200],
                       'SVM__max_iter': [5000],
                       'SVM__degree': [2, 3],
                       'SVM__coef0': [0, 1],
                       'SVM__class_weight': [None, 'balanced']
                       }
