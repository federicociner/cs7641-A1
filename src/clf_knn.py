from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNN(object):

    def __init__(self):
        """ Construct the KNN classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline(
            [('Scale', StandardScaler()), ('KNN', KNeighborsClassifier())])

        # set up parameter grid for parameters to search over
        self.params = {'KNN__metric': ['manhattan', 'minkowski', 'chebyshev'],
                       'KNN__n_neighbors': np.arange(1, 30, 1),
                       'KNN__weights': ['uniform', 'distance'],
                       'KNN__algorithm': ['ball_tree', 'kd_tree'],
                       'KNN__leaf_size': [1, 5, 10, 15, 20, 30, 40]
                       }
