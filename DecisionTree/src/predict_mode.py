import numpy as np


class PredictMode():
    def __init__(self):
        """
        This is a simple classifier that just looks at the targets in the dataset
        and learns to always predict the mode (most common) target.

        For example:
            >>> model = PredictMode()
            >>> model.fit(None, np.array([1, 2, 2, 3, 3, 3]))
            >>> model.predict(None)
            3

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Looking at the 
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of shape (n, d)
                 where n is number of examples and d is number of features.
            targets (np.array): numpy array containing true labels for each of the N
                examples.
        Output:
            None: Just update self.most_common_class with the most common label
        """
        elements, freq = np.unique(targets, return_counts=True)
        self.most_common_class = elements[np.argmax(freq)]
        #raise NotImplementedError

    def predict(self, features):
        """
        Predicts classes for each example in `features` using the trained model.
        Note that for PredictMode, this function won't actually use the values of `features`.

        Args:
            features (np.array): numpy array of shape (n, d)
                 where n is number of examples and d is number of features.
        Outputs:
            predictions (np.array): numpy array of size (n, ) which has the
                predictions for the input data.
        """
        return self.most_common_class
        raise NotImplementedError
