#!/usr/bin/env python

import numpy as np

from activation import sigmoid, identity


def init_network() -> dict:
    """Initialize three-layered neural network."""
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def _forward(z: np.array, W: np.array, b: np.array,
             activation: str) -> np.array:
    """Propagate the signal from the previous to the next layer."""
    a = np.dot(z, W) + b
    if activation == 'sigmoid':
        return sigmoid(a)
    elif activation == 'identity':
        return identity(a)


def forward(network: dict, x: np.array) -> np.array:
    """Propagate the signal from the input to the output."""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    z1 = _forward(x, W1, b1, 'sigmoid')
    z2 = _forward(z1, W2, b2, 'sigmoid')
    y = _forward(z2, W3, b3, 'identity')
    return y
