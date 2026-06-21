import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = np.copy(x)
        for n in range(len(weights)):
            if (n == len(weights)-1):
                h = np.dot(h,weights[n])+biases[n]
                break

            h = np.maximum(0, np.dot(h,weights[n])+biases[n])

        return np.round(h, 5)
