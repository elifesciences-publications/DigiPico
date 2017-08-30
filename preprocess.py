from __future__ import print_function
import numpy as np
import pandas
from sklearn.utils import shuffle
import linecache


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


def generate_data_from_file(filename, feature_size, batch_size, usecols=None, delimiter=',', skiprows=0, dtype=np.float32):
    while 1:
        batch_counter = 0
        if usecols is None:
            usecols = range(1, feature_size+1)
            x_batch = np.zeros([batch_size, feature_size])
            y_batch = np.zeros([batch_size, 1])
        else:
            x_batch = np.zeros([batch_size, len(usecols)])
            y_batch = np.zeros([batch_size, 1])
        with open(filename, 'r') as train_file:
            for line in train_file:
                    batch_counter += 1
                    line = line.rstrip().split(delimiter)
                    y = np.array([dtype(line[0])])
                    x = [dtype(line[k]) for k in usecols]
                    x = np.reshape(x, (-1, len(x)))
                    x_batch[batch_counter - 1] = x
                    y_batch[batch_counter - 1] = y
                    if batch_counter == batch_size:
                        batch_counter = 0
                        # print(x_batch)
                        yield (x_batch, y_batch)


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


def prep_data_all(path_train, over_sample_rate, cols):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # cols = range(1, 66)
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

    # Now add all the rest of 0 and 1s to create the test set
    test_data = data_pos[pe:, :]
    test_data = np.append(test_data, data_neg[ne:, :], axis=0)

    np.savetxt("test.csv", test_data, delimiter=",", fmt='%10.5f')
    np.savetxt("train.csv", train_data, delimiter=",", fmt='%10.5f')

    print("Data Preparation Done!")
    return train_data, test_data


# x_train, y_train, x_test, y_test = prep_data_all('Data/Sahand_All_No-Filter.csv', 1)

def load_preprocessed_data(folder=''):

    # x_train = iter_loadtxt(folder + 'x_train.csv')
    # y_train = iter_loadtxt(folder + 'y_train.csv')
    # x_test = iter_loadtxt(folder + 'x_test.csv')
    # y_test = iter_loadtxt(folder + 'y_test.csv')
    #
    # y_train = np.reshape(y_train, y_train.shape[0], 1)
    # y_test = np.reshape(y_test, y_test.shape[0], 1)
    #
    # return x_train, y_train, x_test, y_test
    train = iter_loadtxt(folder + 'train.csv')
    test = iter_loadtxt(folder + 'test.csv')

    return train, test


if __name__ == "__main__":
    prep_data_all('Sahand_All_No-Filter.csv', 1)
