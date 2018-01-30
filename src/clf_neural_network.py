from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
import numpy as np


class MLP(object):

    def __init__(self):
        """ Construct the multi-layer perceptron classifier object

        """
        # set up model pipeline, scaling training data to have zero mean and
        # unit variance
        self.pipeline = Pipeline([('Scale', StandardScaler()), ('MLP', MLPClassifier(
            random_state=0, max_iter=2000, early_stopping=True))])

        # set up parameter grid for parameters to search over
        self.params = {'MLP__activation': ['logistic', 'relu'],
                       'MLP__alpha': np.arange(0.05, 3, 0.1),
                       'MLP__hidden_layer_sizes': [(32), (64), (128), (32, 64, 32), (64, 128, 64)]
                       }
