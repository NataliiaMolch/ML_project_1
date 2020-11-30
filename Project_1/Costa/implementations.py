# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import random


def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0
    N = y.size

    for k in range(max_iters):
        e = y - tx.dot(w)
        loss = 1/(2*N)*e.dot(e)
        grad_Loss = -1/N*(tx.transpose()).dot(e)
        w = w - gamma*grad_Loss
        print(e.dot(e))

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    w = initial_w
    loss = 0
    N = y.size

    for k in range(max_iters):
        r = list(range(N))
        random.shuffle(r)
        for j in r:
            e = y[j] - tx[j].dot(w)
            loss = 0.5*e*e
            grad_loss = -tx[j]*e
            w = w - gamma*grad_loss
        print(loss)

    return w, loss


def least_squares(y, tx):
    N = y.size
    w = np.linalg.solve(tx, y)
    e = y - tx.dot(w)
    loss = 0.5/N*e.dot(e)

def ridge_regression(y, tx, lambda_):
    M = tx.transpose().dot(tx) + lambda_*np.identity(30)
    w = np.linalg.inv(M).dot(tx.transpose()).dot(y)
    e = y - tx.dot(w)
    N = y.size
    loss = 1 / (2 * N) * e.dot(e) + lambda_*w.dot(w)

    return w, loss


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def logistic_gradient(y, tx, w):

    logistic_gradient = np.sum((sigmoid(w.T @ tx) - y) * tx)

    return logistic_gradient


def logistic_loss(y, tx, w):

    y_len = 1
    return -1. / y_len * np.sum(y * np.log(sigmoid(w.T @ tx)) + (1 - y) * np.log(1 - sigmoid(w.T @ tx)))


def logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma):

    w = initial_w
    loss = 0

    for k in range(max_iters):

        grad_loss = logistic_gradient(y, tx, w)
        w = w - gamma * grad_loss
        loss = logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):

    N, d = tx.shape

    r = list(range(N))
    random.shuffle(r)

    w = initial_w
    loss = 0

    for k in range(max_iters):
        for i in r:
            y_sample = y[i]
            tx_sample = tx[i, :]
            gradient = regularized_logistic_gradient(y_sample, tx_sample, lambda_, w)
            loss = regularized_logistic_loss(y_sample, tx_sample, lambda_, w)
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
    print(reg_logistic_gradient.shape)

    return reg_logistic_gradient


def regularized_logistic_loss(y, tx, lambda_, w):

    loss = np.log(1 + np.exp(np.transpose(tx).dot(w))) - y * np.transpose(tx).dot(w) + lambda_ / 2. * np.linalg.norm(w) ** 2

    return loss


def replace_data(x):
    x_tmp = x
    N, d = x.shape
    for j in range(d):
        mean =  np.mean(x[x[:,j]!=-999,j])
        for i in range(N):
            if(x_tmp[i, j] == -999):
                x[i, j] = mean

    return x_tmp


def normalize_data(tx):
    # normalizing data by features and add bias
    input_data = np.zeros_like(tx, dtype=np.float)
    for i in range(input_data.shape[1]):
        input_data[:, i] = tx[:, i] - np.mean(tx[:, i])
        input_data[:, i] = tx[:, i] / np.std(tx[:, i])
    return np.concatenate((np.ones((tx.shape[0], 1)), input_data), axis=1)


def delete_missing_values(x, y):
    row_nmb = []
    nr = x.shape[0]
    for i in range(nr):
        if np.sum(x[i] == -999.) > 0:
            row_nmb.append(i)
    print("% of axis to be deleted is ", len(row_nmb) / nr)
    return np.delete(x, row_nmb, axis=0), np.delete(y, row_nmb, axis=0)

