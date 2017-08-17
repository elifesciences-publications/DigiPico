from __future__ import print_function

import numpy as np
import keras
import pandas
import nn
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    weights_path = 'mutation_logistic_wts.h5'

    # load dataset
    dataframe = pandas.read_csv('Data/test1.csv', header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 1:].astype(float)
    Y = dataset[:, 0].astype(int)

    # choose a subset
    x_test = X[40:50, :]  # 10 1s
    x_test = np.append(x_test, X[90:100, :], axis=0)  # 10 0s
    y_test = Y[40:50]
    y_test = np.append(y_test, Y[90:100])

    input_dim = X.shape[1]
    nb_classes = 2

    batch_size = 128
    epochs = 1000

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

    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = np.round(np.clip(y_test, 0, 1))
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)

    total = x_test.shape[0]

    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp/total,fp/total,tn/total,fn/total))
