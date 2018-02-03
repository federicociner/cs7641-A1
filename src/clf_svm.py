from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np


class SVM_RBF(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline([('Scale', StandardScaler()), ('SVMR', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVMR__kernel': ['rbf'],
                       'SVMR__C': 10.0 ** np.arange(-3, 8),
                       'SVMR__gamma': 10.0 ** np.arange(-5, 6),
                       'SVMR__cache_size': [4000],
                       'SVMR__max_iter': [30000],
                       'SVMR__class_weight': ['balanced']
                       }


class SVM_PLY(object):

    def __init__(self):
        """ Construct the SVM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline([('Scale', StandardScaler()), ('SVMP', SVC())])

        # set up parameter grid for parameters to search over
        self.params = {'SVMP__kernel': ['poly'],
                       'SVMP__C': 10.0 ** np.arange(-3, 8),
                       'SVMP__gamma': 10.0 ** np.arange(-5, 6),
                       'SVMP__cache_size': [4000],
                       'SVMP__max_iter': [30000],
                       'SVMP__degree': [2, 3],
                       'SVMP__coef0': [0, 1],
                       'SVMP__class_weight': ['balanced']
                       }
