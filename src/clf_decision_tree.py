from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class DT(object):

    def __init__(self):
        """ Construct the KNN classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline([('DT', RandomForestClassifier())])

        # set up parameter grid for parameters to search over
        self.params = {'DT__criterion': ['gini', 'entropy'],
                       'DT__class_weight': ['balanced'],
                       'DT__max_depth': np.arange(1, 20, 3),
                       'DT__min_samples_leaf': np.arange(1, 5, 1),
                       'DT__n_estimators': [100, 200, 500, 1000]
                       }
