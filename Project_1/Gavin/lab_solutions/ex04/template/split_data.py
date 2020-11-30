# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    N = y.shape[0]
    N_train = round(ratio*N)
    # Pick N_train numbers from range(N)
    train_idxs = np.random.choice(range(N), size=N_train, replace=False)
#     print(train_idxs)
    X_train = x[train_idxs]
    y_train = y[train_idxs]
    X_test = [x[index] for index in range(N) if index not in train_idxs]
    y_test = [y[index] for index in range(N) if index not in train_idxs]
    
    return X_train, y_train, X_test, y_test
