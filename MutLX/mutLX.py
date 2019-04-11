""" Main Training Function for MutLX

This module contains the main code for training and evaluating using MutLX.

"""
from __future__ import print_function
import sys
import numpy as np
import pandas
import sklearn
import utils_mutLX
import linecache
import keras
import tensorflow as tf
import random as rn
import subprocess
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import argparse
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle
from sklearn.externals import joblib
from scipy import stats
from keras import backend as K
from tensorflow.python.ops import math_ops
from keras.optimizers import Adam, Nadam
from keras.activations import softmax
from keras.losses import categorical_crossentropy, logcosh


class DropoutPrediction(object):
    """Class used to apply dropouts at test time.

       Proposed by Marcin MoÅ¼ejko

    """
    def __init__(self, model):
        self.f = K.function(
                [model.layers[0].input,
                 K.learning_phase()],
                [model.layers[-1].output])

    def predict(self, x, n_iter=100):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x, 1]))
        result = np.array(result).reshape(n_iter, len(x)).T
        return result


if __name__ == "__main__":

    # Set parameters
    nb_classes = 2
    cols = range(1, 43)
    input_dim = len(cols) - 1

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=8, help='The batch size used for training.')
    parser.add_argument('--epochs', type=int, required=False, default=10, help='The number of epochs to train.')
    parser.add_argument('--sampling_num', type=int, required=False, default=25, help='The number of subsets.')
    parser.add_argument('--drop_it', type=int, required=False, default=100, help='The number of predictions '
                                                                                 'sampled by dropping neurons.')
    parser.add_argument('--pscore_cf', type=int, required=False, default=0.2, help='Cutoff value for '
                                                                                    'probability scores.')
    parser.add_argument('--auc_cf', type=int, required=False, default=0.9, help='Cutoff value for AUC '
                                                                                 'to identify samples with true UTDs.')
    parser.add_argument('--tpr_cf', type=int, required=False, default=0.95, help='The required true positive rate '
                                                                                 'for recovery of true UTDs.')
    parser.add_argument('--input', type=str, required=True, help='Path to CSV file.')
    parser.add_argument('--out_path', type=str, required=False, default='.',
                        help='The path under which to store the output.')
    parser.add_argument('--sample_name', type=str, required=False, default='DigiPico',
                        help='The name of sample.')
    # Parse command line arguments
    args = parser.parse_args()

    input_file = args.input
    path = args.out_path
    sample = args.sample_name
    epochs = args.epochs
    sampling_num = args.sampling_num
    dropout_iter = args.drop_it
    batch_size = args.batch_size
    pscore = args.pscore_cf
    auc = args.auc_cf
    tpr = args.tpr_cf

    # Set tensorboard callback
    tbCallBack = keras.callbacks.TensorBoard(log_dir=path+'/log') 

    # Load data and normalise
    all_set, test_ind, neg_ind, pos_ind, hpos_ind, names = utils_mutLX.prep_typebased(input_file, cols)

    scaler = StandardScaler()
    test_set = all_set[np.sort(np.concatenate((pos_ind,neg_ind)))]
    test_set[:, 1:] = scaler.fit_transform(test_set[:, 1:])
    all_set[:, 1:] = scaler.transform(all_set[:, 1:])

    out1 = all_set[:, 0]
    out2_var = all_set[:, 0]

    for cnt in range(sampling_num):

        # Prepare training subset for level 1 training
        np.random.seed(cnt+1)
        pos_ind_subset = np.random.choice(pos_ind, len(neg_ind), replace=0)
        train_set = all_set[np.sort(np.concatenate((pos_ind_subset, neg_ind)))]

        # Level 1 training and test
        print ("Level 1 training: subset ", cnt+1)
        model_L1 = utils_mutLX.build_model(input_dim, nb_classes - 1, type='ml-binary')
        history = model_L1.fit(train_set[:, 1:], train_set[:, 0],
                               batch_size=batch_size,
                               epochs=epochs,
                               verbose=2,
                               callbacks=[tbCallBack])
        print ("Level 1 test: subset ", cnt+1)
        y_pred = model_L1.predict(test_set[:, 1:])

        # Pruning
        TPs = [ind for ind, label in enumerate(test_set[:, 0]) if label == 1 and y_pred[ind] > 0.3]
        TNs = [ind for ind, label in enumerate(test_set[:, 0]) if label == 0 and y_pred[ind] < 0.7]

        # Prepare training subset for level 2 training
        np.random.seed(cnt+1)
        TPs_subset = np.random.choice(TPs, len(TNs), replace=0)
        train_set = test_set[np.sort(np.concatenate((TPs_subset, TNs)))]

        # Level 2 training and test
        print ("Level 2 training: subset ", cnt+1)
        model_L2 = utils_mutLX.build_model(input_dim, nb_classes - 1, type='ml-binary')
        model_L2.fit(train_set[:, 1:], train_set[:, 0],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=[tbCallBack])
        weights_path = path + '/' + sample + '_logistic_wts.h5'
        model_L2.save_weights(weights_path)

        print("Level 2 test: subset ", cnt+1)

        y_pred = model_L2.predict(all_set[:, 1:])
        out1 = np.column_stack((out1, y_pred))
        model_T2 = utils_mutLX.build_model(input_dim, nb_classes-1, type='ml-binary-dropout', weights_path=weights_path)
        pred_with_dropout = DropoutPrediction(model_T2)
        y_pred = pred_with_dropout.predict(all_set[:, 1:], dropout_iter)
        y_pred_var = stats.describe(y_pred, axis=1)[3]
        out2_var = np.column_stack((out2_var, y_pred_var))

    # Calculate final results and save
    print("Calculate final results")
    y_pred_mean_1 = stats.describe(out1[:, 1:], axis=1)[2]
    y_pred_mvar = stats.describe(out2_var[:, 1:], axis=1)[2]
    final = np.column_stack((y_pred_mean_1, y_pred_mvar))

    # Plot and calculate thresholds
    thr = utils_mutLX.mutLX_plots(final, neg_ind, hpos_ind, pscore, auc, tpr, path+'/'+sample)

    final = np.column_stack((final, np.repeat("PASS",len(y_pred_mvar))))
    final[final[:,1].astype(float)>thr,2] = "FAIL_Uncertain"
    final[final[:,0].astype(float)<=pscore,2] = "FAIL_LowScore"
    save = np.column_stack((names, final))
    header = ['Mutation', 'Type', 'Probability_Score', 'Uncertainty_Score', 'Result']
    pd.DataFrame(save.astype(str)).to_csv(path+'/'+sample+'_scores.csv', header=header, index=None)
    
