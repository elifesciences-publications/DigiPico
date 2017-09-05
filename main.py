from __future__ import print_function
import sys
import numpy as np
import pandas
import nn
import sklearn
from sklearn import linear_model
import preprocess
import linecache
import keras
import subprocess
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, average_precision_score

if __name__ == "__main__":
    weights_path = ''
    # weights_path = 'mutation_logistic_wts.h5'
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    nb_classes = 2
    batch_size = 128
    test_batch_size = 1
    ''' Instead of epochs on the data, we can increase over_sampling rate
    So that in the next epoch, different 0 samples are chosen (but same 1s) '''
    epochs = 40
    over_sampling_rate = 2  # ATTENTION: Check the max in each setting
    # Set tensorboard callback
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./summary/log3')

    # Load Dataset
    train, test = preprocess.prep_data_all('Data/Sahand_OptMap_Chr22.csv', range(1, 67), over_sampling_rate)
    # train, test = preprocess.load_preprocessed_data('')
    test_folder = ''
    train_folder = ''
    train_row_num = subprocess.check_output(['wc', '-l', train_folder + 'train.csv'])
    test_row_num = subprocess.check_output(['wc', '-l', test_folder + 'test.csv'])
    train_size = int(train_row_num.rstrip().split(' ')[0])
    test_size = int(test_row_num.rstrip().split(' ')[0])
    with open(train_folder + 'train.csv', 'r') as data_file:
        line = (data_file.next()).rstrip().split(',')
    input_dim = len(line) - 1  # First col is the label, so skip it in counting the feature numbers
    train_steps_per_epoch = int(train_size/batch_size)
    test_steps_per_epoch = int(test_size/test_batch_size)
    print(train_size, 'train samples')
    print(test_size, 'test samples')

    # # Visualize the data:
    # plt.scatter(train[:, 35:36], train[:, 36:37], c=train[:, 0], s=40, cmap=plt.cm.Spectral)
    # plt.show()
    # # Logistic Regression using scikit
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # clf.fit(train[:, 1:], train[:, 0])
    #
    # # Print accuracy
    # LR_predictions = clf.predict(test[:, 1:])
    # print('Accuracy of logistic regression: %d ' % float(
    #     (np.dot(test[:, 0], LR_predictions) + np.dot(1 - test[:, 0], 1 - LR_predictions))
    #      / float(test[:, 0].size) * 100) +'% ' + "(percentage of correctly labelled datapoints)")
    #
    # weights = np.array(clf.coef_)
    # print(weights[0])

    # Build the model
    model = nn.build_model(input_dim, nb_classes-1, type='binary', weights_path=weights_path)
    # Print a summary of the model
    model.summary()
    # When given a weight path, just go to testing otherwise train.
    if weights_path == '':
        history = model.fit(train[:, 1:], train[:, 0],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1)  # , callbacks=[tbCallBack])

    score = model.evaluate(test[:, 1:], test[:, 0])
    # Generator-based training
    # if weights_path == '':
    #     model.fit_generator(preprocess.generate_data_from_file(train_folder + 'train.csv',
    #         feature_size=input_dim, batch_size=batch_size),
    #         steps_per_epoch=train_steps_per_epoch, nb_epoch=epochs, verbose=1)
    #   #, callbacks=[tbCallBack])
    #
    # score = model.evaluate_generator(preprocess.generate_data_from_file(test_folder + 'test.csv',
    #         feature_size=input_dim, batch_size=batch_size), test_steps_per_epoch)
    #
    # y_pred = model.predict_generator(preprocess.generate_data_from_file(test_folder + 'test.csv',
    #    feature_size=input_dim, batch_size=test_batch_size), test_steps_per_epoch)
    # _, test = preprocess.load_preprocessed_data(test_folder=test_folder, skip_train=True)
    y_pred = model.predict(test[:, 1:])
    y_test = test[:y_pred.shape[0], 0]

    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pred_pos = np.reshape(y_pred_pos, y_pred_pos.shape[0])
    y_pred_neg = np.reshape(y_pred_neg, y_pred_neg.shape[0])

    y_pos = np.round(np.clip(y_test, 0, 1))
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fn = np.sum(y_pos * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)

    total_pos = np.sum(y_pos) + sys.float_info.epsilon
    total_neg = np.sum(y_neg) + sys.float_info.epsilon

    print('TP: {}%, FP: {}%, TN: {}%, FN: {}%'.format(tp / total_pos, fp / total_pos, tn / total_neg, fn / total_neg))
    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp, fp, tn, fn))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test roc auc:', roc_auc)
    print('Test average precision:', average_precision_score(y_test, y_pred))

    # Save model as json and yaml
    json_string = model.to_json()
    open('mutation_logistic_model.json', 'w').write(json_string)
    yaml_string = model.to_yaml()
    open('mutation_logistic_model.yaml', 'w').write(yaml_string)
    # save the weights in h5 format
    model.save_weights('mutation_logistic_wts.h5')
    np.set_printoptions(suppress=True, precision=1)
    for i, val in enumerate(model.layers[0].get_weights()[0]):
        print(str(i+2) + ":" + str(val))

