#!/usr/bin/env python

import numpy as np

from simple_network import init_network, forward


if __name__ == '__main__':
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)
