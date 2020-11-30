# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
#     y = y.values
#     tx = tx.values
    lambda_dash = lambda_ * 2 * len(y)
    w = np.linalg.inv(tx.T @ tx + lambda_dash * np.eye(tx.T.shape[0])) @ tx.T @ y
    loss = 1./(2*len(y)) * np.sum((y - tx@w)**2) + lambda_ * np.linalg.norm(w)**2
    
    return (w, loss)
