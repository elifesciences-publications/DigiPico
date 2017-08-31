from __future__ import print_function

import numpy as np
import pandas
import nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import preprocess
import linecache
import keras

if __name__ == "__main__":
    weights_path = ''
    # weights_path = 'Important_Weights/mutation_logistic_wts.h5'

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    nb_classes = 2
    batch_size = 128
    # Instead of epochs on the data, we can increase over_sampling rate
    # So that in the next epoch, different 0 samples are chosen (but same 1s)
    epochs = 1
    over_sampling_rate = 1  # ATTENTION: MAX 8 in current set

    # Set tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./summary/log3')

    # load dataset
    # x_train, y_train, x_test, y_test = preprocess.prep_data('Data/Sahand_Chr22_No-Filter.csv','Data/Sahand_Chr21_Filter.csv', over_sampling_rate)
    # train_data, test_data = preprocess.prep_data_all('Data/Filter.csv', over_sampling_rate, cols=range(1, 33))
    # train, test = preprocess.load_preprocessed_data('Small_PD/')

    test_folder = 'Large_PD/'
    train_folder = 'Large_PD/'
    # wc -l filename to extract number of rows of csv file
    train_size = 18240
    test_size = 106946
    # train_size = 2800200
    # test_size = 6268829

    with open(train_folder + 'train.csv', 'r') as data_file:
        for i, line_x in enumerate(data_file):
            if i == 0:
                line_x = line_x.rstrip().split(',')
                break

    input_dim = len(line_x) - 1

    train_steps_per_epoch = int(train_size/batch_size)
    test_steps_per_epoch = int(test_size/batch_size)
    # input_dim = x_train.shape[1]
    # steps_per_epoch = int(x_train.shape[0] / batch_size)

    # Extend the data by rotations
    # Convert
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # Normalize
    # scalar = StandardScaler()
    # x_train = scalar.fit_transform(x_train)
    # x_test = scalar.fit_transform(x_test)
    # x_train[11:] = scalar.fit_transform(x_train[11:])
    # x_test[11:] = scalar.fit_transform(x_test[11:])
    #
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    # Build the model
    model = nn.build_model(input_dim, nb_classes-1, type='ml-binary', weights_path=weights_path)

    # Print a summary of the model
    model.summary()
    # history = model.fit(x_train, y_train,
    #                     batch_size=batch_size,
    #                     epochs=epochs,
    #                     verbose=1,
    #                     )#validation_data=(x_test, y_test))  # , callbacks=[tbCallBack])
    if weights_path == '':
        model.fit_generator(preprocess.generate_data_from_file(train_folder + 'train.csv', feature_size=input_dim, batch_size=batch_size),
                            steps_per_epoch=train_steps_per_epoch, nb_epoch=epochs, verbose=1, callbacks=[tbCallBack])

    score = model.evaluate_generator(preprocess.generate_data_from_file(test_folder + 'test.csv', feature_size=input_dim, batch_size=batch_size), test_steps_per_epoch)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1]);

    # y_pred = model.predict_generator(preprocess.generate_data_from_file('test.csv', feature_size=input_dim, batch_size=batch_size), steps_per_epoch)
    # # y_pred = model.predict(x_test)
    # y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    # y_pred_neg = 1 - y_pred_pos
    # y_pred_pos = np.reshape(y_pred_pos, y_pred_pos.shape[0])
    # y_pred_neg = np.reshape(y_pred_neg, y_pred_neg.shape[0])
    #
    # y_pos = np.round(np.clip(y_test, 0, 1))
    # y_neg = 1 - y_pos
    #
    # tp = np.sum(y_pos * y_pred_pos)
    # tn = np.sum(y_neg * y_pred_neg)
    #
    # fn = np.sum(y_pos * y_pred_neg)
    # fp = np.sum(y_neg * y_pred_pos)
    #
    # total_pos = np.sum(y_pos)
    # total_neg = np.sum(y_neg)
    #
    # print('TP: {}%, FP: {}%, TN: {}%, FN: {}%'.format(tp / total_pos, fp / total_pos, tn / total_neg, fn / total_neg))
    # print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp, fp, tn, fn))

    # Save model as json and yaml
    json_string = model.to_json()
    open('mutation_logistic_model.json', 'w').write(json_string)
    yaml_string = model.to_yaml()
    open('mutation_logistic_model.yaml', 'w').write(yaml_string)

    # save the weights in h5 format
    model.save_weights('mutation_logistic_wts.h5')

