# -*- coding: utf-8 -*-
"""Implementations for project 1."""

import numpy as np

from proj1_helpers import predict_labels, predict_01_labels

##########################################################################
##########################################################################
#### PRE PROCESSING FUNCTIONS                     
##########################################################################
##########################################################################

def replace_data(x):
    x_tmp = x
    N, d = x.shape
    for j in range(d):
        mean =  np.mean(x[x[:,j]!=-999,j])
        for i in range(N):
            if(x_tmp[i, j] == -999):
                x[i, j] = mean

    return x_tmp

def replace_data_0(x):
    x_tmp = x
    N, d = x.shape
    for j in range(d):
        mean =  np.mean(x[x[:,j]!=-999,j])
        for i in range(N):
            if(x_tmp[i, j] == -999):
                x[i, j] = 0

    return x_tmp



def normalize_data(tx):
    # normalizing data by features and add bias
    input_data = np.zeros_like(tx, dtype=np.float)
    
    for i in range(input_data.shape[1]):
        max_ = np.max(tx[:,i])
        min_ = np.min(tx[:,i])
        input_data[:, i] = 2. * (tx[:, i] - min_) / (max_ - min_) - 1.0
    return input_data

def standardize_data(tx):
    input_data = np.zeros_like(tx, dtype = np.float)
    for i in range(input_data.shape[1]):
      input_data[:,i] = tx[:,i] -  np.mean(tx[:,i])
      input_data[:,i] = tx[:,i]/np.std(tx[:,i])
    return input_data

def remove_outliers(X, y, threshold=5.0):
    """# Removes outliers from X, y pair"""
    # Assumes that the column already has mean 0 and std 1

    idx = (np.abs(X) < threshold).all(axis=1)
    
    return X[idx], y[idx], idx

def delete_missing_values(x, y):
    row_nmb = []
    nr = x.shape[0]
    for i in range(nr):
        if np.sum(x[i] == -999.) > 0:
            row_nmb.append(i)
    print("% of axis to be deleted is ", len(row_nmb) / nr)
    return np.delete(x, row_nmb, axis=0), np.delete(y, row_nmb, axis=0)

def oversample(x, y, seed=1):
    """ This assumes that the labels are 0 and 1..."""
    signal_idx = np.where(y == 1)[0]
    background_idx = np.where(y == 0)[0]
    
    assert len(signal_idx) < len(background_idx), "You don't need to oversample..."
    
    extra_sample_idx = np.random.choice(signal_idx, size=len(background_idx) - len(signal_idx), replace=True)    

    x = np.vstack((x, x[extra_sample_idx]))
    y = np.vstack((y.reshape(-1,1), y[extra_sample_idx].reshape(-1,1)))
    
    assert len(np.where(y==1)[0]) == len(np.where(y==0)[0]), "The number of signal and background is still not equal"
    
    return x, y, extra_sample_idx

def one_hot_encode(x, col_num):
    """One hot encode and return the matrix"""
    levels = np.array([0., 1., 2., 3.])

    categorical = x[:,col_num].astype(int)

    assert np.all(np.unique(categorical) == levels), "Check that you have the right column!"
    
    OHEC = np.zeros((categorical.size, categorical.max()+1))
    OHEC[np.arange(categorical.size),categorical] = 1

#    OHEC_x = np.c_[np.delete(x, 22, axis=1), OHEC]
    
    normalized_OHEC = normalize_data(OHEC)
    
    return normalized_OHEC
    

##########################################################################
##########################################################################
#### FEATURE SELECTION FUNCTIONS                     
##########################################################################
##########################################################################

def build_polynomial(x, degree):
    """polynomial basis functions for input data x, for j=2 up to j=degree."""
    """we add the constant separately"""
    """x should be a column or a Nx1 matrix"""
    assert degree >= 2, 'Degree must be greater or equal to 2'
    
    feat_mat = np.zeros((len(x), degree-1))
    for i in range(feat_mat.shape[1]):
        feat_mat[:,i] = x**(i+2)
        
    return feat_mat

def polynomial_expansion(df, degree):
    """Add polynomial expansion columns"""
    for i in range(df.shape[1]):
        df = np.hstack((df, build_polynomial(df[:,i], degree)))
    
    return df

##########################################################################
##########################################################################
#### MACHINE LEARNING ALGORITHMS                 
##########################################################################
##########################################################################


def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0
    N = y.size

    for k in range(max_iters):
        e = y - tx.dot(w)
        loss = 1/(2*N)*e.dot(e)
        grad_Loss = -1/N*(tx.transpose()).dot(e)
        w = w - gamma*grad_Loss

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    N = y.size

    for k in range(max_iters):
        r = list(range(N))
        np.random.shuffle(r)
        loss = 0.0
        for j in r:
            e = y[j] - tx[j].dot(w)
            loss += 0.5*e*e/N
            grad_loss = -tx[j]*e
            w = w - gamma*grad_loss.reshape(w.shape)
        

    return w, loss


def least_squares(y, tx):
    N = y.size
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    e = y - tx.dot(w)
    loss = 0.5/N*(e.T).dot(e)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    M = tx.transpose().dot(tx) + lambda_*np.identity(tx.shape[1])
    w = np.linalg.inv(M).dot(tx.transpose()).dot(y)
    e = y - tx.dot(w)
    N = y.size
    loss = 1 / (2 * N) * e.T.dot(e) + lambda_*w.T.dot(w)

    return w, loss


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def logistic_gradient(y, tx, w):
    
    logistic_gradient = ((sigmoid(np.transpose(tx).dot(w)) - y)*tx).reshape(w.shape)

    return logistic_gradient


def logistic_loss(y, tx, w):
    
    return np.log(1 + np.exp(np.transpose(tx).dot(w))) - y * np.transpose(tx).dot(w)

def logistic_regression(y, tx, initial_w, max_iters, gamma):

    N, d = tx.shape

    r = list(range(N))
    np.random.seed(1)
    np.random.shuffle(r)

    w = initial_w

    for k in range(max_iters):
        loss = 0.0
        for i in r:
            y_sample = y[i]
            tx_sample = tx[i, :]
            gradient = logistic_gradient(y_sample, tx_sample, w)
            loss += logistic_loss(y_sample, tx_sample, w)/N
            w = w - gamma * gradient

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    N, d = tx.shape

    r = list(range(N))
    np.random.seed(1)
    np.random.shuffle(r)

    w = initial_w
    
    for k in range(max_iters):
        loss = 0.0
        for i in r:
            y_sample = y[i]
            tx_sample = tx[i, :]
            gradient = regularized_logistic_gradient(y_sample, tx_sample, lambda_, w)
            loss += regularized_logistic_loss(y_sample, tx_sample, lambda_, w)/N
            w = w - gamma * gradient
        print(loss)
    return w, loss


def regularized_logistic_gradient(y, tx, lambda_, w):

    reg_logistic_gradient = ((sigmoid(np.transpose(tx).dot(w)) - y)*tx).reshape(w.shape) + lambda_*w

    return reg_logistic_gradient


def regularized_logistic_gradient2(y, tx, lambda_, w):
    """
    Computes the regularized logistic gradient, assuming SGD so each sample is passed individually
    """
    reg_logistic_gradient = np.sum((sigmoid(w.T @ tx) - y) * tx) + lambda_ * np.sum(w)

    return reg_logistic_gradient


def regularized_logistic_loss(y, tx, lambda_, w):

    loss = np.log(1 + np.exp(np.transpose(tx).dot(w))) - y * np.transpose(tx).dot(w) + lambda_ / 2. * np.linalg.norm(w) ** 2

    return loss

##########################################################################
##########################################################################
#### CROSS VALIDATION FUNCTIONS             
##########################################################################
##########################################################################
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_OLS(X, y, cutoffs, k_fold=4, seed=1):
    """Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    accuracy = np.zeros((k_fold, len(cutoffs)))
    f1 = np.zeros((k_fold, len(cutoffs)))
    
    for k in range(k_fold):
        print("Training on the fold number {} out of {}...".format(k+1, k_fold))
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        w, loss_tr = least_squares(y_train, X_train)
        
        for l in range(len(cutoffs)):
            y_pred = predict_01_labels(w, X_test, cutoffs[l])

            accuracy[k, l] = accuracy_score(y_test, y_pred)
            f1[k, l] = f1_score(y_test, y_pred)

    return np.mean(accuracy, axis=0), np.mean(f1, axis=0)

def cross_validation_SGD(X, y, cutoffs, initial_w, gamma, max_iters, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    # This is going to be a grid search on gamma
    accuracy = np.zeros((k_fold, len(gamma), len(cutoffs)))
    f1 = np.zeros((k_fold, len(gamma), len(cutoffs)))

    for k in range(k_fold):
        print("Training on the fold number {} out of {}...".format(k+1, k_fold))
        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(gamma)):
            w, loss_tr = least_squares_SGD(y = y_train, tx = X_train, initial_w = initial_w, max_iters = max_iters, gamma = gamma[j])
            
            for l in range(len(cutoffs)):
                y_pred = predict_01_labels(w, X_test, cutoffs[l])
                
                accuracy[k, j, l] = accuracy_score(y_test, y_pred)
                f1[k, j, l] = f1_score(y_test, y_pred)
    
    return np.mean(accuracy, axis=0), np.mean(f1, axis=0)

def cross_validation_RR(X, y, cutoffs, lambda_, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    # This is going to be a grid search on lambda_
    accuracy = np.zeros((k_fold, len(lambda_), len(cutoffs)))
    f1 = np.zeros((k_fold, len(lambda_), len(cutoffs)))
    
    for k in range(k_fold):
        print("Training on the fold number {} out of {}...".format(k+1, k_fold))

        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]

        for j in range(len(lambda_)):     
            w, loss_tr = ridge_regression(y = y_train, tx = X_train, lambda_ = lambda_[j])
            
            for l in range(len(cutoffs)):
                y_pred = predict_01_labels(w, X_test, cutoffs[l])
                
                accuracy[k, j, l] = accuracy_score(y_test, y_pred)
                f1[k, j, l] = f1_score(y_test, y_pred)

    return np.mean(accuracy, axis=0), np.mean(f1, axis=0)

def cross_validation_LR(X, y, cutoffs, initial_w, gamma, max_iters, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)
    
    # This is going to be a grid search on gamma
    accuracy = np.zeros((k_fold, len(gamma), len(cutoffs)))
    f1 = np.zeros((k_fold, len(gamma), len(cutoffs)))

    for k in range(k_fold):
        print("Training on the fold number {} out of {}...".format(k+1, k_fold))

        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(gamma)):
            w, loss_tr = logistic_regression(y = y_train, tx = X_train, initial_w = initial_w, max_iters = max_iters, gamma = gamma[j])
            
            for l in range(len(cutoffs)):
                y_pred = predict_01_labels(w, X_test, cutoffs[l])
            
                accuracy[k, j, l] = accuracy_score(y_test, y_pred)
                f1[k, j, l] = f1_score(y_test, y_pred)
            
    return np.mean(accuracy, axis=0), np.mean(f1, axis=0)

def cross_validation_RLR_gamma(X, y, cutoffs, initial_w, gamma, max_iters, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)

    # Try over the gamma with fixed lambda_
    lambda_ = 6E-7
    
    # Search on gamma
    accuracy = np.zeros((k_fold, len(gamma), len(cutoffs)))
    f1 = np.zeros((k_fold, len(gamma), len(cutoffs)))
    
    for k in range(k_fold):
        print("Training on the fold number {} out of {}...".format(k+1, k_fold))

        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(gamma)):    
            w, loss_tr = reg_logistic_regression(y = y_train, tx = X_train, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, gamma = gamma[j])
        
            for l in range(len(cutoffs)):
                y_pred = predict_01_labels(w, X_test, cutoffs[l])
                
                accuracy[k, j, l] = accuracy_score(y_test, y_pred)
                f1[k, j, l] = f1_score(y_test, y_pred)
    
    return np.mean(accuracy, axis=0), np.mean(f1, axis=0)

def cross_validation_RLR_lambda(X, y, cutoffs, initial_w, lambda_, max_iters, k_fold=4, seed=1):
    """# Returns the mean accuracy based on k_fold cross validation"""
    all_indices = build_k_indices(y, k_fold, seed)

    # Try over the lambda (regularisation) with fixed gamma
    gamma = 1e-2
    
    # Search on gamma
    accuracy = np.zeros((k_fold, len(lambda_), len(cutoffs)))
    f1 = np.zeros((k_fold, len(lambda_), len(cutoffs)))
    
    for k in range(k_fold):
        print("Training on the fold number {} out of {}...".format(k+1, k_fold))

        test_indices = all_indices[k]
        train_indices = np.setdiff1d(range(len(y)), test_indices)
        
        y_test = y[test_indices]
        X_test = X[test_indices]
        
        y_train = y[train_indices]
        X_train = X[train_indices]
        
        for j in range(len(lambda_)):    
            w, loss_tr = reg_logistic_regression(y = y_train, tx = X_train, lambda_ = lambda_[j], initial_w = initial_w, max_iters = max_iters, gamma = gamma)
        
            for l in range(len(cutoffs)):
                y_pred = predict_01_labels(w, X_test, cutoffs[l])
                
                accuracy[k, j, l] = accuracy_score(y_test, y_pred)
                f1[k, j, l] = f1_score(y_test, y_pred)
    
    return np.mean(accuracy, axis=0), np.mean(f1, axis=0)


##########################################################################
##########################################################################
#### Accuracy metrics           
##########################################################################
##########################################################################

def accuracy_score(y_test, y_pred):
    """Calculates accuracy as a percentage"""
    y_test = y_test.reshape(y_pred.shape)
    return len(np.where(y_test == y_pred)[0]) / len(y_test) * 100

def f1_score(y_test, y_pred):
    """Assumes labels to be 0 and 1 with 0 being 'negative' and 1 being 'positive"""
    y_test = y_test.reshape(y_pred.shape)
    TP = len(np.where((y_pred == 1.) & (y_test == 1.))[0])
    FP = len(np.where((y_pred == 1.) & (y_test == 0.))[0])
    FN = len(np.where((y_pred == 0.) & (y_test == 1.))[0])
#    TN = len(np.where((y_pred == negative) & (y_test == -1))[0])
    
    return 2 * TP / (2 * TP + FP + FN) * 100