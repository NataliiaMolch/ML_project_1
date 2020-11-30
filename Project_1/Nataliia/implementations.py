#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:02:38 2019

@author: nataliyamolchanova
"""
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import(create_csv_submission, predict_labels, load_csv_data)
#%% DATA PREPROCESSING
def data_processing(x,y):
    x = replace_missing_values(x)
    tx = standartization(x)
    tx = polynomial_expansion(df, degree)
    tx = remove_correlated(tx)
    tx, y = dataset_balancing(x, y , y_value = -1.)
    tx = standartization(tx)
    return tx, y
def normatization(tx):
    input_data = np.zeros_like(tx, dtype = np.float)
    for i in range(input_data.shape[1]):
      minn = np.min(tx[:,i])
      input_data[:,i] = tx[:,i] -  minn
      input_data[:,i] /= (np.max(tx[:,i]) - minn)
    return input_data
def standartization(tx):
    input_data = np.zeros_like(tx, dtype = np.float)
    for i in range(input_data.shape[1]):
      input_data[:,i] = tx[:,i] -  np.mean(tx[:,i])
      input_data[:,i] = tx[:,i]/np.std(tx[:,i])
    return input_data
def remove_outliers(X, y, threshold=5.0):
    idx = (np.abs(X) < threshold).all(axis=1)
    return X[idx], y[idx], idx
def dataset_balancing(x, y , y_value = -1.):
    if len(y[np.where(y == 1.)]) > len(y[np.where(y == y_value)]):
        y1 = y[np.where(y == 1.)]
        y0 = y[np.where(y != 1.)]
        x1 = x[np.whew(y == 1.)]
        x0 = x[np.where(y != 1.)]
    else: 
        y0 = y[np.where(y == 1.)]
        y1 = y[np.where(y != 1.)]
        x0 = x[np.where(y == 1.)]
        x1 = x[np.where(y != 1.)]
    indexes = np.random.permutation(np.arange(len(y0)))
    i = 0
    while(len(y1) != len(y0)):
        y0 = np.append(y0, y0[indexes[i]])
        noize = np.random.normal(0.,0.2, x.shape[1])
        x = np.append(x, x[indexes[i]] + noize)
        i+=1
    x = np.concatenate((x1, x0))
    y = np.concatenate((y1, y0))
    assert x.shape[1] == x1.shape[1], "Concatenation of x is insuccessful"
    assert y.shap[0] == x.shape[0], "Concatenation of y is insuccessful"
    return x, y            
def replace_missing_values(x):
    ncol = x.shape[1]
    for i in range(ncol):
        where = np.where(x[:,i] == -999.)[0]
        filled_values = x[:,i][np.where(x[:,i] != -999.)]
        mean = np.mean(filled_values)
        x[where,i] = mean
    return x
def remove_correlated(x):
    R = np.corrcoef(x.T)
    R = abs(R)
    tmp = (R > 0.98)
    index = []
    for i in range(R.shape[0]):
        for j in range(i):
            if tmp[i,j]:
                index.append(i)

    print("% of features to remove is ", j/x.shape[1])
     
    x = np.delete(x, index ,axis = 1)
     
    return x
 
def build_polynomial(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree."""
    """we add the constant separately"""
    """x should be a column or a Nx1 matrix"""
    assert degree >= 2, 'Degree must be greater or equal to 2'
    
    feat_mat = np.zeros((len(x), degree-1))
    for i in range(feat_mat.shape[1]):
        feat_mat[:,i] = x**(i+2)
#    
    return feat_mat

def polynomial_expansion(df, degree):
    """Add polynomial expansion columns"""
    for i in range(df.shape[1]):
        df = np.hstack((df, build_polynomial(df[:,i], degree)))
    
    return df              
#%% LOSS FUNCTIONS
def MSE(y, tx, w):
    return np.average((y - np.dot(tx,w)) ** 2)
def MSE_long(y, tx, w): #is made because usual computation causes RAM overflow
    mse = 0.
    len_y = len(y)
    for i in range(len_y):
      mse += (y[i] - np.dot(tx[i], w)) **2
    return mse/np.float(len_y)
def MAE(y,tx, w):
    return np.average(np.abs(y - np.dot(tx,w)))
def RMSE(y,tx,w):
    return np.sqrt(2.*MSE_long(y,tx,w))
def loss_function(y,tx,w):
    return MSE_long(y,tx,w)
#%% ADDITIONAL FUNCTIONS FOR ALGORITHMS
    #compute_gradient
    #compute_stoch_gradient
    #batch_iter
    #sigmoid
    #log_loss
    #log_loss_m
    #calculate_stoch_gradient
    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    tmp = np.dot(tx,w)
    tmpp = y + (tmp<0).astype(np.float) - (tmp>=0).astype(np.float)
    return -np.dot(tx.T, tmpp)/float(y.shape[0])
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    tmp = np.dot(tx,w)
    tmpp = y + (tmp<0).astype(np.float) - (tmp>=0).astype(np.float)
    return np.expand_dims(np.average(-np.dot(tx.T, tmpp), axis = 1), axis =1)
def batch_iter(y, tx, batch_size = 10, shuffle=True):
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    num_batches = int(data_size/batch_size)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    tmp = 0.
    print("Calculating log loss")
    for i in range(len(y)):
        tmpp = tx[i] @ w
        tmp += (np.log(1+ np.exp(tmpp)) - y[i]*tmpp)
    return tmp
def log_loss_m(y, tx, w):
    txw = tx @ w
    return np.sum(np.log( np.exp(txw) + 1.) - y.T @ txw )
def calculate_stoch_gradient(y_i, tx_i, w):
    return tx_i.T * (sigmoid(np.dot(tx_i, w)) - y_i)
          
#%% ALGORITHMS
            
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    r = 0.75
    for n_iter in range(max_iters):
        gamma = 1./pow(np.float(1+n_iter), r)
        w = w - gamma*compute_gradient(y, tx, w)
        if gamma <= 1e-5:
          break
        if n_iter%100 == 0:
          print("Making {} iteration, {} iterations remain".format(n_iter, max_iters - n_iter))
    print("DG returnind results, gamma = {}".format(gamma))
    return w,loss_function(y,tx,w)  
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma): #using batch
    """Stochastic gradient descent algorithm."""
    
    w = initial_w
    n_iter = 0
    r = 0.75
    while(n_iter < max_iters):
      for b_y,b_tx in batch_iter(y, tx, batch_size = 1, shuffle=True):
          gamma = 1./pow(np.float(1+n_iter), r)
          w = w - gamma*compute_stoch_gradient(b_y, b_tx, w)
          n_iter+=1
          if n_iter%10000 == 0:
            print( "Iteration #{}, gamma = {}".format(n_iter, gamma))
          if n_iter >= max_iters or gamma <= 1e-5:
              break
    
    print("SDG returnind results, gamma = {}".format(gamma))
    return w,loss_function(y,tx,w)

def least_squares(y, tx):
    w0 = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T, y))
    return w0, loss_function(y,tx,w0)

def ridge_regression(y, tx, lambda_):
    N, d = tx.shape
    w0 = np.dot(np.linalg.inv(np.dot(tx.T, tx) + np.float(2*N*lambda_)*
                              np.ones((d,d), dtype = np.float)), np.dot(tx.T, y))
    return w0, loss_function(y,tx,w0)

def logistic_regression(y, tx, initial_w, max_iters, gamma, hessian = True):
    r = 0.75
    n_iter = 0
    w = initial_w
    y[np.where(y == -1.)] = 0.
    while(n_iter < max_iters):
        for b_y,b_tx in batch_iter(y, tx, batch_size = 1, shuffle=True):
            gamma = 1./pow(np.float(1+n_iter), r)
            if n_iter%10000 == 0:
                print( "Iteration #{}, gamma = {}".format(n_iter, gamma))
            w -= gamma*calculate_stoch_gradient(b_y, b_tx, w)
            n_iter +=1
            if n_iter >= max_iters or gamma <= 1e-5:
                break
        if n_iter >= max_iters or gamma <= 1e-5:
            break
    return w, log_loss(y,tx,w)
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    r = 0.75
    n_iter = 0
    w = initial_w
    y[np.where(y == -1.)] = 0.
    while(n_iter < max_iters):
        for b_y,b_tx in batch_iter(y, tx, batch_size = 1, shuffle=True):
            gamma = 1./pow(np.float(1+n_iter), r)
            w -= gamma*(calculate_stoch_gradient(b_y, b_tx, w) + lambda_*w)
            n_iter +=1
            if n_iter >= max_iters or gamma <= 1e-5:
                break
        if n_iter >= max_iters or gamma <= 1e-5:
            break
    return w, log_loss_m(y,tx,w)

#%% CROSS-VALIDATION
    
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
        
        prediction = predict_labels(w, X_test)

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
            
            prediction = predict_labels(w, X_test)
            
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
            
            prediction = predict_labels(w, X_test)
            
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
            
            prediction = predict_labels(w, X_test, y_value = 0., thrshld = 0.5)
            
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
                
                prediction = predict_labels(w, X_test, y_value = 0., thrshld = 0.5)
                
                accuracy[k, j, l] = len(np.where(y_test == prediction)[0]) / len(y_test) * 100
    
    return gamma, lambda_, np.mean(accuracy, axis=0) 