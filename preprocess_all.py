from __future__ import print_function
import numpy as np
import pandas
from sklearn.utils import shuffle


def prep_data_all(path_train, path_test, over_sample_rate):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    dataframe = pandas.read_csv(path_train, header=None)
    dataset = dataframe.values
    feature_begin = 2
    final_col = int((dataframe.columns[dataframe.isnull().any()].tolist())[0]) # If final col is null, take it out
    feature_end = final_col - 1

    # split into input (X) and output (Y) variables
    X = dataset[:, np.r_[feature_begin:feature_end]].astype(float)  # If used,33: in mlp, reduce neurons
    Y = dataset[:, 1].astype(int)
    # X = np.nan_to_num(X)

    # ASSUME DATA IS SORTED
    positive_num = np.count_nonzero(Y)

    X_pos = X[0:positive_num, :]
    Y_pos = Y[0:positive_num]

    X_neg = X[positive_num:, :]
    Y_neg = Y[positive_num:]

    # Shuffle so that not all are taken from same chromosomes
    X_pos, Y_pos = shuffle(X_pos, Y_pos, random_state=2)
    X_neg, Y_neg = shuffle(X_neg, Y_neg, random_state=3)

    train_ratio = 0.85
    pb = 0
    pe = int(train_ratio * positive_num)
    pnum = pe * over_sample_rate
    nb = 0
    ne = pnum  # If over sample rate = 1, this would be equal to number of 1s

    if ne > Y.shape[0]:  # Just to make sure over sampling rate doesn't exceed the number of zeros
        ne = Y.shape[0]
        print('NE greater than limit!')

    x_train = X_pos[pb:pe, :]
    x_train = np.tile(x_train, (over_sample_rate, 1))  # Over_sample positives
    x_train = np.append(x_train, X_neg[nb:ne, :], axis=0)
    y_train = Y_pos[pb:pe]
    y_train = np.tile(y_train, (over_sample_rate, 1))
    y_train = np.append(y_train, Y_neg[nb:ne])

    # Shuffle the training data so 1 and 0 are not together
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    # Now add all the rest of 0 and 1s to create the test set
    x_test = X_pos[pe:, :]
    y_test = Y_pos[pe:]
    x_test = np.append(x_test, X_neg[ne:, :], axis=0)
    y_test = np.append(y_test, Y_neg[ne:], axis=0)

    x_test, y_test = shuffle(x_test, y_test, random_state=5)

    return x_train, y_train, x_test, y_test





