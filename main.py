from __future__ import print_function

import numpy as np
import keras
import pandas
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


def build_model(input_dim,output_dim,type):

    if type == 'multi-class':

        # MLP with Softmax for multi-class
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])

    elif type == 'binary':
        # Sigmoid used for binary classification, In logistic regression,
        # random weight initialization is not so important
        model = Sequential()
        model.add(Dense(output_dim, input_dim=input_dim, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['binary_accuracy'])

    elif type == 'ml-binary':
        # Sigmoid used for binary classification, In logistic regression,
        # random weight initialization is not so important
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='sigmoid'))

        # Metric: binary accuracy, it calculates K.mean(K.equal(y_true, K.round(y_pred)))
        # Meaning: the mean accuracy rate across all predictions
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['binary_accuracy'])

    return model


# load dataset

# MNIST data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
tbCallBack = keras.callbacks.TensorBoard(log_dir='./summary/log2')

dataframe = pandas.read_csv('Data/test1.csv', header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:, 1:].astype(float)
Y = dataset[:, 0].astype(int)

x_train = X[0:40, :]
x_train = np.append(x_train, X[50:120, :],axis = 0)
y_train = Y[0:40]
y_train = np.append(y_train, Y[50:120])

#x_validate = X[300:360, :]
#y_validate = Y[300:360]

x_test = X[40:50, :]
x_test = np.append(x_test, X[120:, :],axis = 0)
y_test = Y[40:50]
y_test = np.append(y_test, Y[120:])

input_dim = x_train.shape[1]
output_dim = y_train.shape[0]
nb_classes = 2

batch_size = 128
epochs = 1000

# Extend the data by rotations

# Preprocess input data
# When using the Theano backend, you must explicitly declare a dimension for the depth of the input
x_train = x_train.reshape(x_train.shape[0], input_dim)
x_test = x_test.reshape(x_test.shape[0], input_dim)
#x_validate = x_validate.reshape(x_validate.shape[0], input_dim)

# Convert
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_validate= x_validate.astype('float32')

# Normalize
#scalar = StandardScaler()
#x_train = scalar.fit_transform(x_train)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)
#y_validate = keras.utils.to_categorical(y_validate, nb_classes)

# Build the model (MLP)
model = build_model(input_dim, nb_classes, type='binary')

# Print a summary of the model
model.summary()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test), callbacks=[tbCallBack])

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model as json and yaml
json_string = model.to_json()
open('mutation_Logistic_model.json', 'w').write(json_string)
yaml_string = model.to_yaml()
open('mutation_Logistic_model.yaml', 'w').write(yaml_string)

# save the weights in h5 format
model.save_weights('mutation_Logistic_wts.h5')


# Use scikit for training:
# evaluate model with standardized dataset
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasClassifier(build_fn=build_model(input_dim, nb_classes, type='binary'), epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))