# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:17:01 2019

@author: Gavin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from proj1_helpers import predict_labels

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    """
    
    w = initial_w
    
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss))
        
    return (w.values, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, verbose=True):
    """
    Linear regression using stochastic gradient (without replacement)
    """
    
    assert max_iters <= y.shape[0], 'The number of iterations should be less than the number of samples'
    
    # Choose the indexes for which we should sample from
    idxs = np.random.choice(range(y.shape[0]), size=max_iters, replace=False)
        
    w = initial_w
#    import pdb; pdb.set_trace()
    for n_iter, idx in enumerate(idxs):
        # Get the relevant sample
        y_sample = y[idx]
        tx_sample = tx[idx]
        gradient = compute_gradient(y_sample, tx_sample, w)
        loss = compute_loss(y_sample, tx_sample, w)
        w = w - gamma * gradient
        if n_iter%1000 == 0 and verbose:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, ||w|| = {w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w = np.linalg.norm(w)))
        
    return (w, loss)

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    """
#    import pdb; pdb.set_trace()
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = compute_loss(y, tx, w)
    
    return (w, loss)
    
def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations (mse loss)
    """
    lambda_dash = lambda_ * 2 * len(y)
    w = np.linalg.inv(tx.T @ tx + lambda_dash * np.eye(tx.T.shape[0])) @ tx.T @ y
    loss = 1./(2*len(y)) * np.sum((y - tx@w)**2) + lambda_ * np.linalg.norm(w)**2
    
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma, verbose):
    """
    Logistic regression using SGD
    """

    assert max_iters <= y.shape[0], 'The number of iterations should be less than the number of samples'
    
    # Choose the indexes for which we should sample from
    idxs = np.random.choice(range(y.shape[0]), size=max_iters, replace=False)
        
    w = initial_w
#    import pdb; pdb.set_trace()
    losses = []
    
    for n_iter, idx in enumerate(idxs):
        # Get the relevant sample
        y_sample = y[idx]
        tx_sample = tx[idx]
        gradient = compute_logistic_gradient(y_sample, tx_sample, w)
        loss = compute_logistic_loss(y_sample, tx_sample, w)
        
        # For later...
        losses.append(loss)
        
        w = w - gamma * gradient
        
        if n_iter%1000 == 0 and verbose:
            print("Stochastic Gradient Descent for Logistic Regression({bi}/{ti}): loss={l}, ||w|| = {w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w = np.linalg.norm(w)))
    
    plt.plot(losses)
    plt.savefig('plots/losses.png')
    
    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=True):
    """
    Regularized logistic regression using SGD
    """
    assert max_iters <= y.shape[0], 'The number of iterations should be less than the number of samples'
    
    # Choose the indexes for which we should sample from
    idxs = np.random.choice(range(y.shape[0]), size=max_iters, replace=False)
        
    w = initial_w
#    import pdb; pdb.set_trace()
    for n_iter, idx in enumerate(idxs):
        # Get the relevant sample
        y_sample = y[idx]
        tx_sample = tx[idx]
        gradient = compute_regularized_logistic_gradient(y_sample, tx_sample, lambda_, w)
        loss = compute_regularized_logistic_loss(y_sample, tx_sample, lambda_, w)
        w = w - gamma * gradient
        if n_iter%1000 == 0 and verbose:
            print("Stochastic Gradient Descent for Regulariszed Logistic Regression({bi}/{ti}): loss={l}, ||w|| = {w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w = np.linalg.norm(w)))
    
    return (w, loss)

def compute_gradient(y, tx, w, method='mse'):
    """
    Calculate the gradient of the loss using mse or mae (subgradient in this case)
    """
    assert (method in ['mse', 'mae']), 'Method must be mse or mae'
#    import pdb; pdb.set_trace()
    if np.size(y) == 1: # Scalar, one sample
        gradient = -1. * tx.T * (y - tx @ w) if method == 'mse' else -1. * tx.T * np.sign(y - tx@w)
    else: # More samples
        y_len = y.shape[0]
        gradient = -1./y_len * tx.T @ (y - tx@w) if method == 'mse' else -1./y_len * tx.T @ np.sign(y - tx@w)
    
    return gradient

def compute_loss(y, tx, w, method='mse'):
    """
    Calculate the loss either using mse or mae
    """
#    import pdb; pdb.set_trace()
    assert (method in ['mse', 'mae']), 'Method must be mse or mae'
    if np.size(y) == 1: # Scalar, one sample
        loss = 1./2 * np.sum((y - tx@w)**2) if method == 'mse' else np.sum(np.abs(y-tx@w))
    else: # More samples
        y_len = y.shape[0]
        loss = 1./(2*y_len) * np.sum((y.reshape(-1,1) - tx@w.reshape(-1,1))**2) if method == 'mse' else 1./y_len * np.sum(np.abs(y.reshape(-1,1)-tx@w.reshape(-1,1)))
        
    return loss

def compute_logistic_gradient(y, tx, w):
    """
    Calculates the logistic loss gradient, assuming SGD so each sample is passed individually
    """
    
    logistic_gradient = np.sum((sigmoid(w.T @ tx) - y) * tx)
    
    return logistic_gradient

def compute_logistic_loss(y, tx, w):
    """
    Calculates the logistic loss, assuming SGD so each sample is passed individually
    """
    loss = np.sum(np.log(1 + np.exp(w.T @ tx)) - y * w.T @ tx)
    
#    if loss in [float('NaN'), float('Inf')]: import pdb; pdb.set_trace()
    
    return loss

def compute_regularized_logistic_gradient(y, tx, lambda_, w):
    """
    Computes the regularized logistic gradient, assuming SGD so each sample is passed individually
    """
    regularized_logistic_gradient = np.sum((sigmoid(w.T @ tx) - y) * tx) + lambda_ * np.sum(w)
        
    return regularized_logistic_gradient

def compute_regularized_logistic_loss(y, tx, lambda_, w):
    """
    Calculates the regularized logistic loss, assuming SGD so each sample is passed individually
    """
#    y_len = 1
#    return -1./y_len * np.sum( y * np.log(sigmoid(w.T @ tx )) + (1 - y) * np.log(1-sigmoid(w.T @ tx))) + lambda_/2. * np.linalg.norm(w)**2

    loss = np.sum(np.log(1 + np.exp(w.T @ tx)) - y * w.T @ tx) + lambda_/2. * np.linalg.norm(w)**2
    
    return loss

def sigmoid(z):
    """Sigmoid function for logistic regression"""
    return 1./(1+np.exp(-z))

def impute(column, method = 'mean'):
    """ Fill in values which have -999.0"""
    assert (method in ['mean', 'median', 'bootstrap']), 'Method must be either mean, median or bootstrap'
    column = column.to_numpy().copy()
    
    impute_index = np.where(column == -999.)[0]
    
    filled_values = column[np.where(column != -999.)]
    
    if method == 'mean':
        mean = np.mean(filled_values)
        column[impute_index] = mean
        
    elif method == 'median':
        median = np.median(filled_values)
        column[impute_index] = median
        
    elif method == 'bootstrap':
        bootstrap_samples = np.random.choice(filled_values, size=len(impute_index), replace=True)
        np.put(column, impute_index, bootstrap_samples) # This is inplace
        
    return column

def scale(column, method = 'standard'):
    """ Scale values according to two methods (standardise to mean 0, std 1 or minmax so values are between 0 and 1)"""
    assert (method in ['standard', 'minmax']), 'Method must be either standard or minmax'
    
    column = column.to_numpy().copy()

    assert len(np.where(column == -999.)[0]) == 0, 'There are -999s in the data... clean them first!'
    
    if method == 'standard':
        column = (column - np.mean(column))/np.std(column)
    elif method == 'minmax':
        column = (column - np.min(column))/(np.max(column) - np.min(column))
    
    return column

def remove_outliers(X, y, threshold=5.0):
    """# Removes outliers from a dataframe column"""
    # Assumes that the column already has mean 0 and std 1

    idx = (np.abs(X) < threshold).all(axis=1)
    
    return X[idx], y[idx], idx



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):#, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    other_indices = np.setdiff1d(range(len(y)), k_indices)
    
    y_test = y[k_indices]
    tx_test = x[k_indices]
    
    y_train = y[other_indices]
    tx_train = x[other_indices]
    
    w, loss_tr = ridge_regression(y_train, tx_train, lambda_)
    
    loss_te = 1./(2*len(y)) * np.sum((y_test - tx_test@w)**2) + lambda_ * np.linalg.norm(w)**2
        
    prediction = predict_labels(w, tx_test, 0.0)
    
    accuracy = len(np.where(y_test - prediction == 0)[0]) / len(y) * 100
    
    return loss_tr, loss_te, accuracy

def cross_validation_demo(x, y):
    seed = 1
#    degree = 7
    k_fold = 5
    lambdas = np.logspace(-5, 2, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    test_accuracy = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation: TODO
    # ***************************************************
    for lambda_  in lambdas:
        rloss_tr = []
        rloss_te = []
        accuracy = []
        for k in range(k_fold):
            l_tr, l_te, accuracy_fold = cross_validation(y, x, k_indices[k], k_fold, lambda_)#, degree)
            rloss_tr.append(np.sqrt(l_tr))
            rloss_te.append(np.sqrt(l_te))
            accuracy.append(accuracy_fold)
            
        rmse_tr.append(np.mean(rloss_tr))
        rmse_te.append(np.mean(rloss_te))
        test_accuracy.append(np.mean(accuracy))
        
    print("The VARIANCE of the TRAINING RMSE IS {}".format(np.var(rmse_tr)))
    print("The VARIANCE of the TEST RMSE IS {}".format(np.var(rmse_te)))
    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    cross_validation_accuracy(lambdas, test_accuracy)
    
    return rmse_te, test_accuracy

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.figure()
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation_accuracy(lambdas, test_accuracy):
    plt.figure()
    plt.semilogx(lambdas, test_accuracy, marker=".", color='b', label='test accuracy')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation_accuracy")
    
def cross_validation_OLS(X, y, k_fold=4, seed=1):
    """Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    accuracy = np.zeros(k_fold)
    
    for k in range(k_fold):
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        w, loss_tr = least_squares(y_train, X_train)
        
        prediction = predict_labels(w, X_test, 0.0)

        accuracy[k] = len(np.where(y_test == prediction)[0]) / len(y_test) * 100
    
    
    return np.mean(accuracy)

def cross_validation_SGD(X, y, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    # Try over the Gamma (learning rate)
    gamma = np.logspace(-6, -3, 10)
    
    # This is going to be a grid search on gamma
    accuracy = np.zeros((k_fold, len(gamma)))

    for k in range(k_fold):
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(gamma)):
            # Corresponds to 1 'epoch'
            print(gamma[j])
            w, loss_tr = least_squares_SGD(y = y_train, tx = X_train, initial_w = np.random.random(size=X_train.shape[1])*0.01, max_iters = len(y_train), gamma = gamma[j], verbose=False)
            
            prediction = predict_labels(w, X_test, 0.0)
            
            accuracy[k, j] = len(np.where(y_test == prediction)[0]) / len(y_test) * 100
        
    return np.hstack((gamma.reshape(-1,1), np.mean(accuracy, axis=0).reshape(-1,1)))

def cross_validation_RR(X, y, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    # Try over the lambda (regularisation)
    lambda_ = np.logspace(-6, -3, 20)
    
    # This is going to be a grid search on lambda_
    accuracy = np.zeros((k_fold, len(lambda_)))
    
    for k in range(k_fold):
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]

        for j in range(len(lambda_)):
            # Corresponds to 1 'epoch'
            print(lambda_[j])
            
            w, loss_tr = ridge_regression(y = y_train, tx = X_train, lambda_ = lambda_[j])
            
            prediction = predict_labels(w, X_test, 0.0)
            
            accuracy[k, j] = len(np.where(y_test == prediction)[0]) / len(y_test) * 100

    return np.hstack((lambda_.reshape(-1,1), np.mean(accuracy, axis=0).reshape(-1,1)))

def cross_validation_LR(X, y, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    # Try over the Gamma (learning rate)
    gamma = np.logspace(-6, -3, 10)
    
    # This is going to be a grid search on gamma
    accuracy = np.zeros((k_fold, len(gamma)))

    for k in range(k_fold):
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(gamma)):
            # Corresponds to 1 'epoch'
            print(gamma[j])
            w, loss_tr = logistic_regression(y = y_train, tx = X_train, initial_w = np.random.random(size=X_train.shape[1])*0.01, max_iters = len(y_train), gamma = gamma[j], verbose=False)
            
            prediction = predict_labels(w, X_test, 0.0)
            
            accuracy[k, j] = len(np.where(y_test == prediction)[0]) / len(y_test) * 100
        
    return np.hstack((gamma.reshape(-1,1), np.mean(accuracy, axis=0).reshape(-1,1)))

def cross_validation_RLR(X, y, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    # Try over the Gamma (learning rate)
    gamma = np.logspace(-6, -3, 2)
    
    # Try over the lambda (regularisation)
    lambda_ = np.logspace(-6, -3, 5)
    
    # This is going to be a grid search on gamma and lambda_
    accuracy = np.zeros((k_fold, len(gamma), len(lambda_)))

    for k in range(k_fold):
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(gamma)):
            # Corresponds to 1 'epoch'
            print(gamma[j])
            for l in range(len(lambda_)):
                w, loss_tr = reg_logistic_regression(y = y_train, tx = X_train, lambda_ = lambda_[l], initial_w = np.random.random(size=X_train.shape[1])*0.01, max_iters = len(y_train), gamma = gamma[j], verbose=False)
                
                prediction = predict_labels(w, X_test, 0.0)
                
                accuracy[k, j, l] = len(np.where(y_test == prediction)[0]) / len(y_test) * 100
    
    return gamma, lambda_, np.mean(accuracy, axis=0) 