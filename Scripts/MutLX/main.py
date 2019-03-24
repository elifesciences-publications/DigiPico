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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle
from sklearn.externals import joblib

# import hyperio as hy

from keras.optimizers import Adam, Nadam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh

#p = {'lr': (2, 10, 30),
#     'first_neuron':[4, 8, 16, 32, 64, 128],
#     'batch_size': [2, 3, 4],
#     'epochs': [5000],
#     'dropout': (0, 0.40, 10),
#     'optimizer': [Adam, Nadam]}


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Not enough arguments. Please enter sample name, analysis path, Epochs, training ratio, and number of columns")
    print("Sample:", sys.argv[1])
    sample = sys.argv[1]
    pth = sys.argv[2]
    print("Epoch:", sys.argv[3])
    print("Training Ratio:", sys.argv[4])

    weights_path = ''
    # weights_path = 'mutation_logistic_wts_FOREXPERIMENTS.h5'
    # fix random seed for reproducibility
    seed = 10
    np.random.seed(seed)
    nb_classes = 2
    batch_size = 8
    test_batch_size = 1
    ''' Instead of epochs on the data, we can increase over_sampling rate
    So that in the next epoch, different 0 samples are chosen (but same 1s) '''
    epochs = int(sys.argv[3])
    over_sampling = False
    cutoff_thr = -0.3  # When 0, probabilities higher than 0.5 are labelled as 1. when -0.3, probabilities higher than
    #  0.8 are considered as 1.
    model_type = 'ml-binary'
    # Set tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir=pth+'/log')

    # Load Dataset
    # Range is from 1 because the very first colomn is Chrom names.
    # Then I cut it in the preprocessing.
    cols = range(1, int(sys.argv[5]))
    train, test = preprocess.prep_data_all(pth + "/train/data/" + sample + ".train.csv", pth + "/train/", cols, sys.argv[4], over_sampling)
    test_folder = pth + "/train/"
    train_folder = pth + "/train/"
    train_row_num = subprocess.check_output(['wc', '-l', train_folder + 'train.csv'])
    test_row_num = subprocess.check_output(['wc', '-l', test_folder + 'test.csv'])
    train_size = int(train_row_num.decode().rstrip().split(' ')[0])
    test_size = int(test_row_num.decode().rstrip().split(' ')[0])
    with open(train_folder + 'train.csv', 'r') as data_file:
        line = (data_file.readline()).rstrip().split(',')
    input_dim = len(line) - 1  # First col is the label, so skip it in counting the feature numbers
    train_steps_per_epoch = int(train_size/batch_size)
    test_steps_per_epoch = int(test_size/test_batch_size)
    print(train_size, 'train samples')
    print(test_size, 'test samples')

    # Normalize Data (Both train and test)
    # Important: Normalize test using same mui and sigma from train
    scaler = StandardScaler()
    train[:, 1:] = scaler.fit_transform(train[:, 1:])
    joblib.dump(scaler, test_folder + "mean_var")
    test[:, 1:] = scaler.transform(test[:, 1:])
    # Visualize the data:
    # plt.scatter(train[:, 35:36], train[:, 36:37], c=train[:, 0], s=40, cmap=plt.cm.Spectral)
    # plt.show()
    '''
    # Logistic Regression using scikit
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(train[:, 1:], train[:, 0])

    # Print accuracy
    LR_predictions = clf.predict(test[:, 1:])
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(test[:, 0], LR_predictions) + np.dot(1 - test[:, 0], 1 - LR_predictions))
         / float(test[:, 0].size) * 100) +'% ' + "(percentage of correctly labelled datapoints)")

    weights = np.array(clf.coef_)
    print(weights[0])
    '''
    # Build the model
    model = nn.build_model(input_dim, nb_classes-1, type=model_type, weights_path=weights_path)
    # Print a summary of the model
    model.summary()
    # Early stopping
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    # Shuffle the labels for a sanity check
    # train[:, 0] = shuffle(train[:, 0], random_state=3)
    # When given a weight path, just go to testing otherwise train.
    if weights_path == '':
        history = model.fit(train[:, 1:], train[:, 0],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            callbacks=[tbCallBack])
                            #validation_split=0.5)

    score = model.evaluate(test[:, 1:], test[:, 0])

    # Save model as json and yaml
    json_string = model.to_json()
    open(train_folder + 'mutation_logistic_model.json', 'w').write(json_string)
    yaml_string = model.to_yaml()
    open(train_folder + 'mutation_logistic_model.yaml', 'w').write(yaml_string)
    # save the weights in h5 format
    model.save_weights(train_folder + 'mutation_logistic_wts.h5')

    '''
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
    '''
    y_pred = model.predict(test[:, 1:])
    y_test = test[:y_pred.shape[0], 0]

    # P(x>= 0.5) = 1 , P(x<0.5) = 0
    y_pred_pos = np.round(y_pred + cutoff_thr)
    y_pred_neg = 1 - y_pred_pos
    y_pred_pos = np.reshape(y_pred_pos, y_pred_pos.shape[0])
    y_pred_neg = np.reshape(y_pred_neg, y_pred_neg.shape[0])

    y_pos = y_test  # np.round(y_test)
    y_neg = 1 - y_pos

    # Number of agreements between y_pos and y_pred_pos is tp
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fn = np.sum(y_pos * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)

    total_pos = np.sum(y_pos) + sys.float_info.epsilon
    total_neg = np.sum(y_neg) + sys.float_info.epsilon
    # print('TP: {}%, FP: {}%, TN: {}%, FN: {}%'.format(tp / total_pos, fp / total_pos, tn / total_neg, fn / total_neg))
    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp, fp, tn, fn))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test roc auc:', roc_auc)
    # print('Test roc threshold:', threshold)
    print('Test average precision:', average_precision_score(y_test, y_pred))

    # Compute the intercross of threshold and rates
    done=0
    i=0
    while done != 1:
        i+=1
        if thresholds[i] < tpr[i]:
            print('Optimal_Threshold:', (thresholds[i]+old_thr)/2)
            done=1
        else:
            old_thr=thresholds[i]

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 0.1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold', color='r')
    ax2.set_ylim([thresholds[-1], 1])
    ax2.set_xlim([fpr[0], 0.1])
    plt.savefig(train_folder + sample + ".ROC.train.10.png")

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold', color='r')
    #ax2.set_ylim([thresholds[-1], thresholds[0]])
    ax2.set_ylim([thresholds[-1], 1])
    ax2.set_xlim([fpr[0], fpr[-1]])
    plt.savefig(train_folder + sample + ".ROC.train.100.png")



