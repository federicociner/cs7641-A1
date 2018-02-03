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
        self.pipeline = Pipeline([('Scale', StandardScaler()), ('MLP', MLPClassifier(max_iter=5000, early_stopping=True))])

        # set up parameter grid for parameters to search over
        alphas = [10 ** -exp for exp in np.arange(-1, 8, 0.5)]
        d = 50
        hidden_layer_size = [(h,) * l for l in [1, 2, 3]
                             for h in [d // 2, d, d * 2]]

        self.params = {'MLP__activation': ['logistic', 'relu'],
                       'MLP__alpha': alphas,
                       'MLP__hidden_layer_sizes': hidden_layer_size
                       }
