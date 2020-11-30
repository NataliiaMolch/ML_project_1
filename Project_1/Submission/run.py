##########################################################################
#### Importing functions
##########################################################################

import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import(
        create_csv_submission, predict_labels, load_csv_data, predict_01_labels)

from implementations import (
        replace_data, normalize_data, remove_outliers, oversample, one_hot_encode
        ,polynomial_expansion
        ,least_squares_GD, least_squares_SGD, least_squares
        ,ridge_regression, logistic_regression, reg_logistic_regression
        ,cross_validation_OLS, cross_validation_SGD
        ,cross_validation_RR
        ,cross_validation_LR, cross_validation_RLR_gamma, cross_validation_RLR_lambda
        )

##########################################################################
#### Loading data
##########################################################################

# load data from train set
_y_train, _tX_train, ids_train = load_csv_data("train.csv")

# change [-1, 1] labels to [0, 1]
y_train = _y_train/2 + 0.5

##########################################################################
#### Data pre-processing
##########################################################################

# replace -999 values with the mean of the other ones
tX_train = replace_data(_tX_train)

# Get the one-hot-encoded columns for later
one_hot_columns = one_hot_encode(tX_train, 22)

# normalize data to std 1 and 0 mean
tX_train = normalize_data(tX_train)

# Add a polynomial expansion - get rid of the one-hot-encoded columns and add them back later
# We do not add the polynomial expansion of the one-hot-encoded columns
# Choose the degree here
polynomial_deg = 5
tX_train = polynomial_expansion(np.delete(tX_train, 22, axis=1), polynomial_deg)

# One-hot-encoding for the integer column
tX_train = np.c_[np.delete(tX_train, 22, axis=1), one_hot_columns]

# Get rid of the outliers from the training set
#tX_train, y_train, ids = remove_outliers(tX_train, y_train, 12.0)

# Oversample the training to get an equal proportion of 0s and 1s in the training set
#tX_train, y_train, ids = oversample(tX_train, y_train)


##########################################################################
#### Cross validation
##########################################################################
N, d = tX_train.shape

# initial weights randomly generated with fixed seed
np.random.seed(1)
w0 = 0.1*np.random.rand(d,1)

# Epochs for the SGD methods
max_iters = 20
# Range of gammas to cross validate on
gamma = np.logspace(-7, -1, 10)
print("gamma cross val : ", gamma)
# Range of lambdas to cross validate on
lambda_ = np.logspace(-10, -3, 9)

# We also try a cross validation of the accuracy with respect to the cut off
# This is to try to compensate for the imbalance in the labels
cutoffs = np.linspace(-.1,1,10)

# uncomment to cross-validate
# Cross validation over both gamma and lambda - the gamma one first
# Assumes a lambda of 2E-3
#RLR_avg_test_accuracy_gamma, _ = cross_validation_RLR_gamma(tX_train, y_train, cutoffs, initial_w = w0, gamma = gamma, max_iters = max_iters)
#print(RLR_avg_test_accuracy_gamma)

# Assumes a gamma of 1E-3
#RLR_avg_test_accuracy_lambda, _ = cross_validation_RLR_lambda(tX_train, y_train, cutoffs, initial_w = w0, lambda_ = lambda_, max_iters = max_iters)
#print(RLR_avg_test_accuracy_lambda)



# We should have the optimal method, parameters and cutoff from the above
##########################################################################

##########################################################################
#### Running the functions
##########################################################################

# parameters for training
gamma = 5e-3
lambda_ = 6e-7

#w, loss = least_squares(y_train, tX_train)

#w, loss = least_squares_GD(y_train, tX_train, initial_w = w0, max_iters = max_iters, gamma = gamma)

#w, loss = least_squares_SGD(y_train, tX_train, w0, max_iters = max_iters, gamma = gamma)

#w, loss = ridge_regression(y_train, tX_train, lambda_)

#w, loss = logistic_regression(y_train, tX_train, initial_w = w0, max_iters = max_iters, gamma = gamma)

w, loss = reg_logistic_regression(y_train, tX_train, lambda_ = lambda_, initial_w= w0, max_iters = max_iters, gamma= gamma)


##########################################################################
#### Calculate the train accuracy
##########################################################################

N = y_train.size
# TRAIN test accuracy for sanity
n_err = len(np.where(y_train != predict_01_labels(w, tX_train, 0.5).reshape(y_train.shape))[0])

print("train accuracy :", 1 - n_err/N)


##########################################################################
#### Generate a prediction on the test set
##########################################################################

# load data from test set
_, _tX_test, ids_test = load_csv_data("test.csv")

# Do the corresponding steps for tX_test (we do not need to remove outliers or oversample, even if we did it on the training set)
tX_test = replace_data(_tX_test)
one_hot_columns = one_hot_encode(tX_test, 22)
tX_test = normalize_data(tX_test)
tX_test = polynomial_expansion(np.delete(tX_test, 22, axis=1), polynomial_deg)
tX_test = np.c_[np.delete(tX_test, 22, axis=1), one_hot_columns]

#predict the labels with 0 threshold
y_pred = predict_labels(w, tX_test, 0.0)

name = "my_submission.csv"

#save predictions
create_csv_submission(ids_test, y_pred, name)
