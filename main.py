from __future__ import print_function

import numpy as np
import keras
import pandas
import nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

if __name__ == "__main__":
    weights_path = ''
    # weights_path = 'mutation_logistic_wts.h5'

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # Set tensorboard callback
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='./summary/log3')

    # load dataset
    # use_colomns = [i for i in range(0, 33)]
    #dataframe = pandas.read_csv('Data/Filter_NoSort.csv', header=None, usecols=use_colomns)
    dataframe = pandas.read_csv('Data/Filter.csv', header=None)
    dataset = dataframe.values

    # MNIST data, shuffled and split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # split into input (X) and output (Y) variables
    X = dataset[:, 2:].astype(float)
    Y = dataset[:, 1].astype(int)

    X = np.nan_to_num(X)

    over_sample_rate = 1
    x_train = X[0:9120]  # 9120 1s
    x_train = np.tile(x_train, (over_sample_rate, 1))
    x_train = np.append(x_train, X[10731:(10731+over_sample_rate*9120)], axis=0)  # 9120 0s
    y_train = Y[0:9120]
    y_train = np.tile(y_train, (over_sample_rate, 1))
    y_train = np.append(y_train, Y[10731:(10731+over_sample_rate*9120)])


    # Shuffle the training data
    x_train, y_train = shuffle(x_train, y_train, random_state = 0)

    #x_validate = X[300:360, :]
    #y_validate = Y[300:360]

    x_test = X[9120:10731]  # 15% of the other ones
    x_test = np.append(x_test, X[(10731+9120):(10731+2*9120)], axis=0)  # 1the other zeros
    y_test = Y[9120:10731]
    y_test = np.append(y_test, Y[(10731+9120):(10731+2*9120)])

    input_dim = x_train.shape[1]
    nb_classes = 2

    batch_size = 128
    epochs = 2 

    # Extend the data by rotations

    # Preprocess input data
    # When using the Theano backend, you must explicitly declare a dimension for the depth of the input
    # x_train = x_train.reshape(x_train.shape[0], input_dim)
    # x_test = x_test.reshape(x_test.shape[0], input_dim)
    #x_validate = x_validate.reshape(x_validate.shape[0], input_dim)

    # Convert
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    #x_validate= x_validate.astype('float32')

    # Normalize
    # scalar = StandardScaler()
    # x_train = scalar.fit_transform(x_train)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)
    #y_validate = keras.utils.to_categorical(y_validate, nb_classes)

    # Build the model (MLP)
    model = nn.build_model(input_dim, nb_classes, type='binary', weights_path=weights_path)

    # Print a summary of the model
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))#, callbacks=[tbCallBack])

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

    # Weighted Over_Sampling for a re-train:
    # Find the false-negatives and sample from them:
    # fn_array = y_test[:, 1] * y_pred[:, 0]

    # Save model as json and yaml
    json_string = model.to_json()
    open('mutation_logistic_model.json', 'w').write(json_string)
    yaml_string = model.to_yaml()
    open('mutation_logistic_model.yaml', 'w').write(yaml_string)

    # save the weights in h5 format
    model.save_weights('mutation_logistic_wts.h5')


    # Use scikit for training:
    # evaluate model with standardized dataset
    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(build_fn=build_model(input_dim, nb_classes, type='binary'), epochs=100, batch_size=5, verbose=0)))
    # pipeline = Pipeline(estimators)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(pipeline, X, Y, cv=kfold)
    # print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
