from __future__ import print_function
import numpy as np
import pandas
from sklearn.utils import shuffle


def prep_data(path_train, path_test, over_sample_rate):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    dataframe = pandas.read_csv(path_train, header=None)
    dataset = dataframe.values
    feature_begin = 2
    feature_end = 65
    binary_flag = 14
    print(dataframe.columns[dataframe.isnull().any()].tolist())
    # split into input (X) and output (Y) variables
    X = dataset[:, np.r_[feature_begin:feature_end]].astype(float)  # If used,33: in mlp, reduce neurons
    Y = dataset[:, 1].astype(int)
    # X = np.nan_to_num(X)

    # ASSUME DATA IS SORTED
    positive_num = np.count_nonzero(Y)
    positive_num_train = int(1.0 * positive_num)

    pb = 0
    pe = positive_num_train
    pnum = positive_num_train * over_sample_rate
    nb = positive_num
    ne = nb + pnum  # in order to have same number of negatives as positives

    if ne > Y.shape[0]:
        ne = Y.shape[0]
        print('NE greated than limit!')

    # pb_test = pe
    # pe_test = nb
    # nb_test = ne
    # ne_test = nb_test + (pe_test - pb_test)  # in order to have same number of positives

    x_train = X[pb:pe, :]  # 9120 1s
    x_train = np.tile(x_train, (over_sample_rate, 1))  # Over_sample positives
    x_train = np.append(x_train, X[nb:ne, :], axis=0)  # 9120 0s
    y_train = Y[pb:pe]
    y_train = np.tile(y_train, (over_sample_rate, 1))
    y_train = np.append(y_train, Y[nb:ne])

    # Shuffle the training data
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    dataframe_test = pandas.read_csv(path_test, header=None)
    dataset_test = dataframe_test.values

    # split into input (X) and output (Y) variables
    x_test = dataset_test[:, np.r_[feature_begin:feature_end]].astype(float)  # If used,33: in mlp, reduce neurons
    # x_test = dataset_test[:, np.r_[5:15]].astype(float)  # If used,33: in mlp, reduce neurons
    y_test = dataset_test[:, 1].astype(int)

    # x_test = np.nan_to_num(x_test)

    #
    # # Take out the test data
    # x_test = X[pb_test:pe_test, :]  # 15% of the other ones
    # x_test = np.append(x_test, X[nb_test:ne_test, :], axis=0)  # 1the other zeros
    # y_test = Y[pb_test:pe_test]
    # y_test = np.append(y_test, Y[nb_test:ne_test])

    return x_train, y_train, x_test, y_test





