# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    feat_mat = np.ones((len(x), degree + 1))
    for i in range(feat_mat.shape[0]):
        for j in range(1,feat_mat.shape[1]):
            feat_mat[i,j] = x[i]**j
    
    return feat_mat