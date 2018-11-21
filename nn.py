import numpy as np
import keras
import keras.layers.advanced_activations as activations
from keras.optimizers import Adam

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.optimizers import RMSprop, SGD
from keras import regularizers


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
        model.add(Dense(input_dim, kernel_initializer='normal', activation='relu', input_shape=(input_dim,)))
        #model.add(Dropout(0.3))
        model.add(Dense(int(input_dim/2), kernel_initializer='normal', activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))

        if weights_path:
            model.load_weights(weights_path)

        # Metric: binary accuracy, it calculates K.mean(K.equal(y_true, K.round(y_pred)))
        # Meaning: the mean accuracy rate across all predictions
        # opt = Adam()
        # model.compile(loss='binary_crossentropy',
        #               optimizer=opt,
        #               metrics=['binary_accuracy'])

    elif type == 'cnn':
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=1, padding='same', activation='relu', input_shape=(input_dim,1)))
        # model.add(MaxPooling1D(pool_size=2))
        #model.add(Dropout(0.2))
        # model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(output_dim, activation='sigmoid'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    elif type == 'car':
        model = Sequential()
        # First layer.
        model.add(Dense(
            64, init='lecun_uniform', input_shape=(input_dim,)
        ))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        # Second layer.
        model.add(Dense(64, init='lecun_uniform'))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(output_dim, kernel_initializer='lecun_uniform', activation='sigmoid'))

        if weights_path:
            model.load_weights(weights_path)

        # Metric: binary accuracy, it calculates K.mean(K.equal(y_true, K.round(y_pred)))
        # Meaning: the mean accuracy rate across all predictions
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['binary_accuracy'])

    return model
