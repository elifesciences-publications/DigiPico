""" Auxilary methods used in MutLX

This module contains functions used in MutLX to build the models, pre-process the data, and plot the results.

   - `iter_loadtxt` is used for a faster line by line reading of large csv files.


   - `prep_typebased` reads and preprocesses data.


   - `build_model` contains the code to build main models used in this version.

   - `mutLX_plots` is used for plotting the outputs

"""

from __future__ import print_function
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
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
from lmfit.models import *
mod = GaussianModel()


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


def mutLX_plots(final, neg_ind, hpos_ind, out):

    data = final[neg_ind]
    #fit Gaussia model for train subset analysis
    x=data[:,0]
    y=data[:,1]
    datax = np.column_stack((x,y))
    datax = datax[datax[:,0].argsort()]

    # calculate the height of fitted normal model as an estimate of true UTD frequency
    pars = mod.guess(datax[:,1], x=datax[:,0])
    result = mod.fit(datax[:,1], pars, x=datax[:,0])
    h = result.params['height'].value

    plt.figure()
    plt.scatter(data[:,0], data[:,1], c='c', s=10, alpha=0.3)
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.005, 0.15])
    plt.plot(datax[:,0], result.best_fit, 'r-')
    plt.xlabel('Probability Score')
    plt.ylabel('Variance in probability values')
    plt.savefig(out + "_train_subset_analysis.png")

    data = data[data[:,2]>0,]
    plt.figure()
    plt.scatter(data[:,2], data[:,3]/data[:,2], c='c', s=10, alpha=0.3)
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.005, 0.7])
    plt.xlabel('Mean of dropout probability values')
    plt.ylabel('Stdev of dropout probability values')
    plt.savefig(out + "_test_dropout_analysis.png")

    # Dispersion of adjusted merge dropouts as a measure of model uncertainty
    data = data[data[:,3]/data[:,2]<0.2]
    data = data[data[:,3]/data[:,2]>0.1]
    adjM = data[:,2] + data[:,3]/data[:,2]

    # Mean of drop-out variances also as a measure of model uncertainty
    data = final[final[:,0]>0.7,]
    data = data[data[:,0]<0.9,]
    MV=np.percentile(data[:,4],30)

    data = final[neg_ind]
    data = data[data[:,0]>0.9,]
    stat = stats.describe(data[:,4])

    h95_OK = 0
    data = final[hpos_ind]
    hG95 = np.percentile(data[:,4],95)
    data = final[neg_ind]
    data = data[data[:,0]>0.9,]
    data = data[data[:,4]<hG95,]
    if np.percentile(data[:,4],25) < 0.025:
        h95_OK = 1

    if h >= 0.05:
        msthr = 0.8
    elif h <= 0.04:
        msthr = 0.7
    else:
        msthr = 10 * h + 0.3

    if np.percentile(adjM,10) >= 0.8:
        minScore = 0.8
    elif np.percentile(adjM,10) <= 0.6:
        minScore = 0.2
    else:
        minScore = 3 * np.percentile(adjM,10) - 1.6

    if MV < 0.1:
    # This suggests a likely high frequency true UTD
        if np.percentile(adjM,10) < msthr:
            thr = 0.2
        else:
            minScore = 0.95
            thr = 0.01
    elif h95_OK == 1:
        minScore = 0.9
        thr = hG95
    else:
        minScore = 0.8
        if h > 0.085:
            thr = 0.2
        elif h > 0.07:
            thr = 0.1
        elif h > 0.06:
            thr = 1.5*h-0.08
        else:
            thr = 0.01

    if np.percentile(data[:,4],25) > 0.05 or (np.percentile(data[:,4],25) > 0.04 and stat[4] < 0):
        thr = 0.09

    data = final[hpos_ind]
    if np.percentile(data[:,4],80) > thr:
        thr = np.percentile(data[:,4],80)
        minScore = 0.99

    thr_adaptive = [minScore, thr]

    mpl.rcParams['axes.linewidth'] = 3

    data = final[neg_ind]
    b="grey"
    plt.figure()
    plt.scatter(data[:,4], data[:,0], c=b, s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([thr, thr], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_UTD_adaptive.png")

    data = final[hpos_ind]
    np.random.shuffle(data)
    b='b'
    plt.figure()
    plt.scatter(data[:1000,4], data[:1000,0], c=b, s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([thr, thr], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_Germline_adaptive.png")

    thr=np.percentile(data[:,4],90)
    minScore=0.8
    thr_tpr90 = [minScore, thr]

    data = final[neg_ind]
    b="grey"
    plt.figure()
    plt.scatter(data[:,4], data[:,0], c=b, s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([thr, thr], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_UTD_tpr90.png")

    data = final[hpos_ind]
    np.random.shuffle(data)
    b='b'
    plt.figure()
    plt.scatter(data[:1000,4], data[:1000,0], c=b, s=20, alpha=0.3)
    plt.xlim([-0.01, 0.21])
    plt.ylim([-0.01, 1.02])
    plt.plot([-0.01, 0.22], [minScore, minScore], 'r--', linewidth=2)
    plt.plot([thr, thr], [-0.2, 1.1], 'r--', linewidth=2)
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Probability Score')
    plt.savefig(out + "_Germline_tpr90.png")

    return thr_tpr90, thr_adaptive
