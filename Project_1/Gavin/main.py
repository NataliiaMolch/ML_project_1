# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:18:58 2019

@author: Gavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sb

from proj1_helpers import(
        create_csv_submission, predict_labels, load_csv_data)

from implementations import (
                least_squares_GD, least_squares_SGD, least_squares
                ,ridge_regression, logistic_regression
                ,reg_logistic_regression
                ,impute, scale, remove_outliers, polynomial_expansion
                ,cross_validation_demo
                ,cross_validation_OLS
                ,cross_validation_SGD
                ,cross_validation_RR
                ,cross_validation_LR
                ,cross_validation_RLR
                )

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%%
features = train.columns.to_list()

features = list(set(features).difference(list(['Id','Prediction'])))

#%%#%% Pre-processing Y should be 0 1 for logistic, can be -1 1 for the rest
#y_train, X_train, ids_train = load_csv_data("train.csv")
#y_test, X_test, ids_test = load_csv_data("test.csv")

y_train = train['Prediction'].map({'s': 1, 'b': -1}).to_numpy()

X_train = train[features]
# One hot encode the PRI_jet_num column
PRI_jet_num = X_train.PRI_jet_num
OHEC = np.zeros((PRI_jet_num.size, PRI_jet_num.max()+1))
OHEC[np.arange(PRI_jet_num.size),PRI_jet_num] = 1
for j in range(OHEC.shape[1]):
    OHEC[:,j] = (OHEC[:,j] - OHEC[:,j].mean())/OHEC[:,j].std()

X_train = X_train.drop('PRI_jet_num', axis=1).apply(impute, method='mean').apply(scale, method='standard')#.apply(remove_outliers, threshold=5)
X_train, y_train, idx = remove_outliers(X_train, y_train, threshold=6)

#X_train = polynomial_expansion(X_train, degree=4)

# Add back the OHEC
X_train = np.c_[X_train, OHEC[idx]]

X_test = test[features]
PRI_jet_num = X_test.PRI_jet_num
OHEC = np.zeros((PRI_jet_num.size, PRI_jet_num.max()+1))
OHEC[np.arange(PRI_jet_num.size),PRI_jet_num] = 1
for j in range(OHEC.shape[1]):
    OHEC[:,j] = (OHEC[:,j] - OHEC[:,j].mean())/OHEC[:,j].std()
X_test = X_test.drop('PRI_jet_num', axis=1).apply(impute, method='mean').apply(scale, method='standard')

#X_test = polynomial_expansion(X_test, degree=4)

# Add back the OHEC
X_test = np.c_[X_test, OHEC]
# Don't remove the outliers from the test...

num_features = X_train.shape[1]

#%% Cross validation

avg_test_accuracy_OLS = cross_validation_OLS(X_train, y_train, k_fold=4, seed=1)

# Cross validation over gamma
avg_test_accuracy_SGD = cross_validation_SGD(X_train, y_train, k_fold=4, seed=1)

# Cross validation over lambda
avg_test_accuracy_RR = cross_validation_RR(X_train, y_train, k_fold=4, seed=1)

# Cross validation over gamma
avg_test_accuracy_LR = cross_validation_LR(X_train, y_train, k_fold=4, seed=1)

# Cross validation over both gamma and lambda
g, l, avg_test_accuracy_RLR = cross_validation_RLR(X_train, y_train, k_fold=4, seed=1)

#%% Testing functions
#np.random.seed(42)

gamma = 0.2
lambda_ = 4E-5

w, loss = least_squares(y = y_train, tx = X_train)
#
w, loss = least_squares_SGD(y = y_train, tx = X_train, initial_w = np.random.random(size=num_features)*0.01, max_iters = 200000, gamma = gamma)
#
w, loss = ridge_regression(y = y_train, tx = X_train, lambda_ = lambda_)
#
w, loss = logistic_regression(y = y_train, tx = X_train, initial_w = np.random.random(size=num_features)*10, max_iters = 125000, gamma = gamma)
#
w, loss = reg_logistic_regression(y = y_train, tx = X_train, lambda_ = lambda_, initial_w = np.random.random(size=num_features)*0.01, max_iters = 200000, gamma = gamma)

plt.plot(w)

#%% Predictive step
y_test = X_test @ w

plt.hist(y_test, bins=200)

y_pred = predict_labels(w, X_test)

#%% Create submission

create_csv_submission(test.Id, y_pred, 'submission.csv')
