from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class DT(object):

    def __init__(self):
        """ Construct the decision tree classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('DT', DecisionTreeClassifier())])

        # set up parameter grid for parameters to search over
        self.params = {'DT__criterion': ['gini', 'entropy'],
                       'DT__class_weight': [None, 'balanced'],
                       'DT__max_depth': np.arange(1, 200, 1),
                       'DT__min_samples_leaf': np.arange(1, 10, 1)
                       }
