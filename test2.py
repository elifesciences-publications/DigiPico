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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle


if __name__ == "__main__":
    weights_path = ''
    weights_path = 'mutation_logistic_wts.h5'
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    nb_classes = 2
    batch_size = 64
    test_batch_size = 1
    ''' Instead of epochs on the data, we can increase over_sampling rate
    So that in the next epoch, different 0 samples are chosen (but same 1s) '''
    epochs = 5000
    over_sampling_rate = 1  # ATTENTION: Check the max in each setting
    cutoff_thr = -0.1  # When 0, probabilities higher than 0.5 are labelled as 1. when -0.3, probabilities higher than
    # 0.8 are considered as 1.
    model_type = 'ml-binary'
    # Set tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./summary/log3')

    # Load Dataset
    # Range is from 1 because the very first colomn is Chrom names.
    # Then I cut it in the preprocessing.
    cols = range(1, 20)
    # cols = [2,3,5,6,7,8,11,12,13,14,15,16,18,19,21,22,23,24,25,26,27,28,29,30,32,33,36,37,38,39,40,41,42,43,44,45,46]
    test = preprocess.prep_test('Data/Test.Data.csv', cols, over_sampling_rate)
    # train, test = preprocess.load_preprocessed_data('')

    # Build the model
    input_dim = 18
    model = nn.build_model(input_dim, nb_classes-1, type=model_type, weights_path=weights_path)
    # Print a summary of the model
    model.summary()
    score = model.evaluate(test[:, 1:], test[:, 0])

    y_pred = model.predict(test[:, 1:])

    np.savetxt("predictions.csv", y_pred, fmt='%10.5f')



