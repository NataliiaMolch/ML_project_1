# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    
    mse = 1./(2*len(y)) * np.sum((y-tx@w)**2)
    return mse, w