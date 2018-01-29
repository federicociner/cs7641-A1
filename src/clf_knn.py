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
            [('Scale', StandardScaler()), ('clf', KNeighborsClassifier())])

        # set up parameter grid for parameters to search over
        self.params = {'clf__metric': ['manhattan', 'euclidean', 'chebyshev'],
                       'clf__n_neighbors': np.arange(1, 55, 3),
                       'clf__weights': ['uniform', 'distance']}
