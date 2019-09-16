#!/usr/bin/env python

import numpy as np


def step(x: np.array, thres: float = 0.0) -> np.array:
    """Return the value of stepu function following the threshold provided."""
    y = x >= thres
    return y.astype(np.int)


def sigmoid(x: np.array) -> np.array:
    """Return the value of sigmoid function."""
    return 1 / (1 + np.exp(-x))


def relu(x: np.array) -> np.array:
    """Return the value of ReLU function."""
    return np.maximum(0, x)


if __name__ == '__main__':
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    print(relu(x))
