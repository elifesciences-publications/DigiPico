from __future__ import print_function

import numpy as np
import keras
import pandas
import nn
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    weights_path = 'mutation_logistic_wts.h5'

    # load dataset
    use_names = list(range(0, 740))
    dataframe = pandas.read_csv('Data/Filter.csv', header=None, names=use_names)
    # dataframe = pandas.read_csv('Data/Filter_NoSort.csv', header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 2:].astype(float)
    Y = dataset[:, 1].astype(int)

    X = np.nan_to_num(X)

    # choose a subset
    # x_test = X[40:50, :]  # 10 1s
    # x_test = np.append(x_test, X[90:100, :], axis=0)  # 10 0s
    # y_test = Y[40:50]
    # y_test = np.append(y_test, Y[90:100])
    x_test = X
    y_test = Y
    # x_test = X[9120:10731, :]  # 15% of the other ones
    # x_test = np.append(x_test, X[(10731 + 9120):, :], axis=0)  # 1the other zeros
    # y_test = Y[9120:10731]
    # y_test = np.append(y_test, Y[(10731 + 9120):])

    input_dim = X.shape[1]
    nb_classes = 2

    # Preprocess input data
    # When using the Theano backend, you must explicitly declare a dimension for the depth of the input
    x_test = x_test.reshape(x_test.shape[0], input_dim)
    # Convert
    x_test = x_test.astype('float32')

    # # Normalize
    # scalar = StandardScaler()
    # x_test = scalar.fit_transform(x_test)

    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    # Build the model & load the weights
    model = nn.build_model(input_dim, nb_classes, type='ml-binary', weights_path=weights_path)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Manually calculate FN,FP,TN,TP:
    y_pred = model.predict(x_test)

    y_pred = np.round(np.clip(y_pred, 0, 1))

    tp = np.sum(y_test[:, 1] * y_pred[:, 1])
    tn = np.sum(y_test[:, 0] * y_pred[:, 0])

    total_pos = np.sum(y_test[:, 1])
    total_neg = np.sum(y_test[:, 0])

    fp = total_pos - tp
    fn = total_neg - tn

    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp/total_pos, fp/total_pos, tn/total_neg, fn/total_neg))