from __future__ import print_function

import numpy as np
import keras
import pandas
import nn

if __name__ == "__main__":
    weights_path = 'mutation_logistic_wts.h5'

    # load dataset
    dataframe = pandas.read_csv('Data/test1.csv', header=None)
    dataset = dataframe.values

    # split into input (X) and output (Y) variables
    X = dataset[:, 1:].astype(float)
    Y = dataset[:, 0].astype(int)

    input_dim = X.shape[1]
    output_dim = X.shape[0]
    nb_classes = 2

    batch_size = 128
    epochs = 1000

    # Extend the data by rotations

    # Preprocess input data
    # When using the Theano backend, you must explicitly declare a dimension for the depth of the input
    x_test = X.reshape(X.shape[0], input_dim)
    # Convert
    x_test = x_test.astype('float32')

    # Normalize
    #scalar = StandardScaler()
    #x_test = scalar.fit_transform(x_test)

    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    y_test = keras.utils.to_categorical(Y, nb_classes)
    #y_validate = keras.utils.to_categorical(y_validate, nb_classes)

    # Build the model & load the weights
    model = nn.build_model(input_dim, nb_classes, type='binary', weights_path=weights_path)

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
