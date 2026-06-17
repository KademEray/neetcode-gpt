import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        mod_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        binary_ce = np.mean(-(y_true * np.log(mod_y_pred) + (1 - y_true) * np.log(1-mod_y_pred)))
        return np.round(binary_ce, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        mod_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        cat_c_e = - (np.sum(y_true * np.log(mod_y_pred)) / len(y_true))
        return np.round(cat_c_e, 4)
