from __future__ import print_function
import numpy as np
import pandas
from sklearn.utils import shuffle


def prep_data(path_train, path_test, over_sample_rate):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # use_colomns = [i for i in range(0, 33)]
    # dataframe = pandas.read_csv(path, header=None, usecols=use_colomns)
    dataframe = pandas.read_csv(path_train, header=None)
    dataset = dataframe.values
    feature_begin = 2
    final_col = int((dataframe.columns[dataframe.isnull().any()].tolist())[0])
    feature_end = final_col - 1
    binary_flag = 14
    # split into input (X) and output (Y) variables
    X = dataset[:, np.r_[feature_begin:feature_end]].astype(float)  # If used,33: in mlp, reduce neurons
    # X = dataset[:, np.r_[5:15]].astype(float)  # If used,33: in mlp, reduce neurons
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


def prep_data_all(path_train, over_sample_rate):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # dataframe = pandas.read_csv(path_train, header=None, dtype=float, usecols=range(1, 53))
    # dataset = dataframe.values

    dataset = np.genfromtxt(path_train, delimiter=',', dtype=float, usecols=range(1, 53))
    feature_begin = 1
    # final_col = int((dataframe.columns[dataframe.isnull().any()].tolist())[0])  # If final col is null, take it out
    feature_end = 51 # final_col - 1

    # split into input (X) and output (Y) variables
    X = dataset[:, np.r_[feature_begin:feature_end]].astype(float)  # If used,33: in mlp, reduce neurons
    Y = dataset[:, 0].astype(int)
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

    np.savetxt("x_train.csv", x_train, delimiter=",")
    np.savetxt("y_train.csv", y_train, delimiter=",")
    np.savetxt("x_test.csv", x_test, delimiter=",")
    np.savetxt("y_test.csv", y_test, delimiter=",")

    return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = prep_data_all('Data/Sahand_All_No-Filter.csv', 1)
def iter_loadtxt(filename, usecols=None, delimiter=',', skiprows=0, dtype=np.float32):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                if usecols is not None:
                    line = [line[i]for i in usecols]
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def prep_data_all_2(path_train, over_sample_rate):

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    cols = range(1, 67)
    # cols = range(1, 33)
    # dataset = np.loadtxt(path_train, delimiter=',', dtype=float, usecols=range(1, 33))
    dataset = iter_loadtxt(path_train, usecols=cols)
    print("Loading Data Done!")
    feature_begin = 1
    feature_end = 51  # final_col - 1

    positive_num = np.count_nonzero(dataset[:, 0])

    data_pos = dataset[0:positive_num, :]
    data_neg = dataset[positive_num:, :]

    # Shuffling to put different chromosomes together
    data_pos = shuffle(data_pos, random_state=3)
    data_neg = shuffle(data_neg, random_state=6)

    train_ratio = 0.85
    pb = 0
    pe = int(train_ratio * positive_num)
    pnum = pe * over_sample_rate
    nb = 0
    ne = pnum  # If over sample rate = 1, this would be equal to number of 1s

    if ne > dataset.shape[0]:  # Just to make sure over sampling rate doesn't exceed the number of zeros
        ne = dataset.shape[0]
        print('NE greater than limit!')

    train_data = data_pos[pb:pe, :]
    train_data = np.tile(train_data, (over_sample_rate, 1))
    train_data = np.append(train_data, data_neg[nb:ne, :], axis=0)

    # Shuffle the training data so 1 and 0 are not together
    train_data = shuffle(train_data, random_state=0)

    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    # Now add all the rest of 0 and 1s to create the test set
    test_data = data_pos[pe:, :]
    test_data = np.append(test_data, data_neg[ne:, :], axis=0)

    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    np.savetxt("x_train.csv", x_train, delimiter=",")
    np.savetxt("y_train.csv", y_train, delimiter=",")
    np.savetxt("x_test.csv", x_test, delimiter=",")
    np.savetxt("y_test.csv", y_test, delimiter=",")
    print("Data Preparation Done!")
    return x_train, y_train, x_test, y_test


# x_train, y_train, x_test, y_test = prep_data_all('Data/Sahand_All_No-Filter.csv', 1)

def load_preprocessed_data():

    x_train = iter_loadtxt('x_train.csv')
    y_train = iter_loadtxt('y_train.csv')
    x_test = iter_loadtxt('x_test.csv')
    y_test = iter_loadtxt('y_test.csv')

    y_train = np.reshape(y_train, y_train.shape[0], 1)
    y_test = np.reshape(y_test, y_test.shape[0], 1)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    prep_data_all_2('Sahand_All_No-Filter.csv.csv', 1)