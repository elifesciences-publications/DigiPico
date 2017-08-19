import numpy as np
import keras
import pandas

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


def build_model(input_dim,output_dim,type,weights_path):

    if type == 'multi-class':

        # MLP with Softmax for multi-class
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='softmax'))

        if weights_path:
            model.load_weights(weights_path)

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

    elif type == 'binary':
        # Sigmoid used for binary classification, In logistic regression,
        # random weight initialization is not so important
        model = Sequential()
        model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid'))

        if weights_path:
            model.load_weights(weights_path)

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

    elif type == 'ml-binary':
        # Sigmoid used for binary classification, In logistic regression,
        # random weight initialization is not so important
        model = Sequential()
        model.add(Dense(70, kernel_initializer='normal', activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.5))
        # model.add(Dense(15, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))

        if weights_path:
            model.load_weights(weights_path)

        # Metric: binary accuracy, it calculates K.mean(K.equal(y_true, K.round(y_pred)))
        # Meaning: the mean accuracy rate across all predictions
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

    return model
