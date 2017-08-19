from __future__ import print_function
import numpy as np
import pandas
from sklearn.utils import shuffle


def prep_data(path, over_sample_rate):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # use_colomns = [i for i in range(0, 33)]
    # dataframe = pandas.read_csv(path, header=None, usecols=use_colomns)
    dataframe = pandas.read_csv(path, header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 2:].astype(float)  # If used,33: in mlp, reduce neurons
    Y = dataset[:, 1].astype(int)

    X = np.nan_to_num(X)

    x_train = X[0:9120, :]  # 9120 1s
    if over_sample_rate > 1:
        x_train = np.tile(x_train, (over_sample_rate, 1))
    x_train = np.append(x_train, X[10731:(10731 + over_sample_rate * 9120), :], axis=0)  # 9120 0s
    y_train = Y[0:9120]
    if over_sample_rate > 1:
        y_train = np.tile(y_train, (over_sample_rate, 1))
    y_train = np.append(y_train, Y[10731:(10731 + over_sample_rate * 9120)])

    # Shuffle the training data
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    # Take out the test data
    test_num = 10731 - 9210
    x_test = X[9120:10731, :]  # 15% of the other ones
    x_test = np.append(x_test, X[(10731 + over_sample_rate * 9120):(10731 + over_sample_rate * 9120 + test_num), :],
                       axis=0)  # 1the other zeros
    y_test = Y[9120:10731]
    y_test = np.append(y_test, Y[(10731 + over_sample_rate * 9120):(10731 + over_sample_rate * 9120 + test_num)])

    return x_train, y_train, x_test, y_test





