from __future__ import print_function
import numpy as np
import pandas
from sklearn.utils import shuffle
import linecache


def prep_data_all(path_train, cols, over_sample_rate):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    dataset = iter_loadtxt(path_train, usecols=cols)
    print("Loading Data Done!")
    positive_num = np.count_nonzero(dataset[:, 0])
    data_pos = dataset[0:positive_num, :]
    data_neg = dataset[positive_num:, :]

    # Shuffling to put different chromosomes together
    data_pos = shuffle(data_pos, random_state=3)
    data_neg = shuffle(data_neg, random_state=6)
    # Since out training data is big enough, we can use a higher train ratio (Ng)
    train_ratio = 0.40
    pb = 0
    pe = int(train_ratio * positive_num)
    pnum = int(pe * over_sample_rate)
    nb = 0
    ne = pnum  # If over sample rate = 1, ne would be equal to number of 1s
    # Make sure over sampling rate doesn't exceed the number of zeros
    if ne > dataset.shape[0]:
        ne = dataset.shape[0]
        print('Oversampling rate greater than limit!')

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


def load_preprocessed_data(train_folder='', test_folder='', skip_train=False, skip_test=False):

    # x_train = iter_loadtxt(folder + 'x_train.csv')
    # y_train = iter_loadtxt(folder + 'y_train.csv')
    # x_test = iter_loadtxt(folder + 'x_test.csv')
    # y_test = iter_loadtxt(folder + 'y_test.csv')
    #
    # y_train = np.reshape(y_train, y_train.shape[0], 1)
    # y_test = np.reshape(y_test, y_test.shape[0], 1)
    #
    # return x_train, y_train, x_test, y_test
    train = []
    test = []
    if not skip_train:
        train = iter_loadtxt(train_folder + 'train.csv')
    if not skip_test:
        test = iter_loadtxt(test_folder + 'test.csv')

    return train, test


if __name__ == "__main__":
    prep_data_all('Sahand_All_No-Filter.csv', 1)
