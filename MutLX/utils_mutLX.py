""" Auxilary methods used in MutLX

This module contains functions used in MutLX to build the models, pre-process the data, and plot the results.

   - `iter_loadtxt` is used for a faster line by line reading of large csv files.


   - `prep_typebased` reads and preprocesses data.


   - `build_model` contains the code to build main models used in this version.

   - `mutLX_plots` is used for plotting the outputs

"""

from __future__ import print_function
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.utils import shuffle
import linecache
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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
import seaborn as sns


def iter_loadtxt(filename, usecols=None, delimiter=',', skiprows=0, dtype=np.float32):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                if usecols is not None:
                    line = [line[i]for i in usecols]
                for item in line:
                    yield item
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def prep_typebased(path_full, cols):

    print("Start Loading Data!")

    dataset = iter_loadtxt(path_full, usecols=cols, dtype='S20')
    names = iter_loadtxt(path_full, usecols=range(2), dtype='S20')

    print("Loading Data Done!")
    print("Start Preparing Data!")

    snp_unq_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type =='SNP-Unq'.encode()]
    snp_hm_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == 'SNP-Hm'.encode()]
    snp_ht_h_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == 'SNP-Ht-H'.encode()]
    snp_ht_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == 'SNP-Ht'.encode()]
    snp_sm_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == 'SNP-Somatic'.encode()]

    neg_ind = snp_unq_ind
    pos_ind = snp_ht_h_ind + snp_ht_ind
    test_ind = pos_ind + neg_ind
    pos_ind = np.sort(pos_ind)
    test_ind = np.sort(test_ind)
    rest_pos_ind = snp_sm_ind + snp_hm_ind
    hpos_ind = snp_ht_h_ind + snp_hm_ind

    # Replacing the names with labels
    dataset[pos_ind, 0] = 1
    dataset[neg_ind, 0] = 0
    dataset[rest_pos_ind, 0] = 1

    dataset = np.float64(dataset)

    print("Data Preparation Done!")
    return dataset, test_ind, neg_ind, pos_ind, hpos_ind, names


def build_model(input_dim, output_dim, type, weights_path=''):

    np.random.seed(7)
    if type == 'ml-binary':
        model = Sequential()
        model.add(Dense(input_dim, kernel_initializer='normal', activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.0))
        model.add(Dense(int(input_dim/2), kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.0))
        model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))
        if weights_path:
            model.load_weights(weights_path)
        # Metric: binary accuracy, it calculates K.mean(K.equal(y_true, K.round(y_pred)))
        # Meaning: the mean accuracy rate across all predictions
        opt = Adam()
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['binary_accuracy'])

    elif type == 'ml-binary-dropout':
        model = Sequential()
        model.add(Dense(input_dim, kernel_initializer='normal', activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(0.8))
        model.add(Dense(int(input_dim/2), kernel_initializer='normal', activation='relu'))
        model.add(Dropout(0.7))
        model.add(Dense(output_dim, kernel_initializer='normal', activation='sigmoid'))
        if weights_path:
            model.load_weights(weights_path)
        # Metric: binary accuracy, it calculates K.mean(K.equal(y_true, K.round(y_pred)))
        # Meaning: the mean accuracy rate across all predictions
        opt = Adam()
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['binary_accuracy'])
    return model


def mutLX_plots(final, neg_ind, hpos_ind, minScore, auc_cf, tpr_cf, out):

    neg_set = final[neg_ind]
    pos_set = final[hpos_ind] 
    neg_set = neg_set[neg_set[:,0]>minScore]
    pos_set = pos_set[pos_set[:,0]>minScore]
    thr = []
    tpr = []
    utd = []
    for t in np.arange(0.25, -0.01, -0.01):
        thr.append(t)
        utd.append(len(neg_set[neg_set[:,1]<=t,1]))
        tpr.append(len(pos_set[pos_set[:,1]<=t,1])/len(pos_set[:,1]))
    fpr = np.array(utd[:])/utd[0]
    roc_auc = auc(fpr, tpr)

    cf = 0
    if roc_auc > auc_cf:
        # Compute the intersection of threshold and rates
        i=0
        while cf == 0:
            if 1-4*thr[i] > tpr[i]:
                cf = (thr[i]+old_thr)/2
            else:
                old_thr=thr[i]
            i += 1
    else:
        i = len(tpr) - 1
        while cf == 0:
            if tpr[i] >= tpr_cf:
                cf = (thr[i]+old_thr)/2
            else:
                old_thr=thr[i]
            i -= 1    

    mpl.rcParams['axes.linewidth'] = 3

    data = np.column_stack((final, np.repeat("SNP",len(final[:,0]))))
    data[neg_ind,2] = "UTDs"
    data[hpos_ind,2] = "Germline SNPs"
    SNPs=["UTDs","Germline SNPs"]
    plt.figure()
    for SNP in SNPs:
        subset = data[data[:,2] == SNP]
        sns.distplot(subset[:,0].astype(float),hist=False,kde = True,kde_kws={'shade': True, 'linewidth': 3,'clip': (0,1)},label=SNP)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    plt.ylim([-0.1, ymax])
    plt.xlim([-0.02, 1.02])
    plt.xlabel('Probability score')
    plt.ylabel('Density')
    plt.plot([minScore, minScore], [0, ymax+2], 'r--', linewidth=2)
    plt.legend(loc=1,bbox_to_anchor=(0.96, 1))
    plt.savefig(out + "_Probability_Score.png")

    plt.figure()
    plt.plot(utd, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=3)
    plt.xlim([0.0, np.amax(utd)])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Number of passed UTDs (False Positives)')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", prop={'size': 18})
    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(utd, thr, markeredgecolor='r', linestyle='dashed', color='r', linewidth=3)
    ax2.set_ylabel('Drop-out variance threshold', color='r')
    ax2.set_ylim([0.25,0])
    ax2.set_xlim([0.0, np.amax(utd)])
    plt.savefig(out + "_ROC.png")

    data = final[neg_ind]
    plt.figure()
    plt.scatter(data[:,1], data[:,0], c='grey', s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([cf, cf], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_UTDs.png")

    data = final[hpos_ind]
    np.random.shuffle(data)
    plt.figure()
    plt.scatter(data[:1000,1], data[:1000,0], c='b', s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([cf, cf], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_Germline.png")

    return cf
