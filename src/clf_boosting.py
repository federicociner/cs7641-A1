from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


class ADA(object):

    def __init__(self):
        """ Construct the AdaBoost classifier object

        """
        # created pre-pruned tree as base estimator
        base_estimator = DecisionTreeClassifier(max_depth=15, min_samples_leaf=3)

        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('ADA', AdaBoostClassifier(base_estimator=base_estimator))])

        # set up parameter grid for parameters to search over
        self.params = {'ADA__learning_rate': np.arange(0.1, 1.0, 0.05),
                       'ADA__n_estimators': [100, 300, 500, 700, 900],
                       }
