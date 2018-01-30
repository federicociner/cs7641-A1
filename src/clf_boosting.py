from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class GBM(object):

    def __init__(self):
        """ Construct the GBM classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('GBM', GradientBoostingClassifier())])

        # set up parameter grid for parameters to search over
        self.params = {'GBM__loss': ['deviance'],
                       'GBM__learning_rate': np.arange(0.1, 1.0, 0.05),
                       'GBM__n_estimators': [100, 300, 500],
                       'GBM__max_depth': np.arange(1, 15, 2),
                       'GBM__min_samples_leaf': np.arange(1, 5, 1)
                       }
