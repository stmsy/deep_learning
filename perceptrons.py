#!/usr/bin/env python

import numpy as np


def _and(x1: int, x2: int,  w1: float = 0.5, w2: float = 0.5,
             b: float = -0.7) -> int:
    """Evaluate the AND gate as single perceptron."""
    if np.dot([x1, x2], [w1, w2]) + b >= 0:
        return 1
    else:
        return 0


def _or(x1: int, x2: int,  w1: float = 0.5, w2: float = 0.5,
            b: float = -0.2) -> int:
    """Evaluate the OR gate as single perceptron."""
    if np.dot([x1, x2], [w1, w2]) + b >= 0:
        return 1
    else:
        return 0


def _nand(x1: int, x2: int,  w1: float = -0.5, w2: float = -0.5,
              b: float = 0.7) -> int:
    """Evaluate the NAND gate as single perceptron."""
    if np.dot([x1, x2], [w1, w2]) + b >= 0:
        return 1
    else:
        return 0
