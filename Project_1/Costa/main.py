import numpy as np

from proj1_helpers import(
        create_csv_submission, predict_labels, load_csv_data)

from implementations import (
                least_squares_GD, least_squares_SGD, least_squares
                ,ridge_regression, logistic_regression
                , normalize_data, delete_missing_values, replace_data,reg_logistic_regression
                )

#load data from train set
y, tX, ids = load_csv_data("train.csv")

# change [-1, 1] labels to [0, 1]
y = y/2 + 0.5

N,d = tX.shape
#initial weights randomly generated
w0 = 10*np.random.rand(d+1,1)

# remplace -999 values with the mean of the other ones
tX = replace_data(tX)
# normalize data to std 1 and 0 mean
tX = normalize_data(tX)

w, L = reg_logistic_regression(y, tX, lambda_ = 0.001, initial_w= w0, max_iters = 10, gamma= 5e-7)

y_pred = predict_labels(w, tX, 0.5)

N = y_pred.size

# accuracy test on train set for sanity
n_err = 0

for i in range(N):
    if (y_pred[i] != y[i]):
        n_err = n_err + 1

print("train accuracy :", n_err/N)

print("weight values")
print(w)

#load data from test set
yb_test, input_data_test, ids_test = load_csv_data('test.csv')

#same data proprecessing
input_data_test = replace_data(input_data_test)
input_data_test = normalize_data(input_data_test)

name = "sub_logistic_sgd"

y_pred = predict_labels(w, input_data_test, 0.5)

#save predictions
create_csv_submission(ids_test, y_pred, name)
