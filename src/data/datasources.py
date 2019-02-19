import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale


class SKLearnDigits():
    def __init__(self):
        self.digits = load_digits()
        self.X = scale(self.digits.data)
        self.y = self.digits.target

        self.n_samples, n_features = self.data.shape
        self.n_digits = len(np.unique(self.digits.target))

        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (self.n_digits, self.n_samples, n_features))
