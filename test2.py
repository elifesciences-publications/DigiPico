from __future__ import print_function
import numpy as np
import nn
import preprocess
import keras
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Not enough arguments. Please enter Directory, Filename and Threshold")
    print("Directory:", sys.argv[1])
    path = sys.argv[1]
    print("Filename:", sys.argv[2])
    filename = sys.argv[2]
    print("Threshold:", sys.argv[3])
    cutoff_thr = 0.5 - float(sys.argv[3])

    weights_path = ''
    weights_path = 'mutation_logistic_wts.h5'
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    nb_classes = 2
    # cutoff_thr = -0.1  # When 0, probabilities higher than 0.5 are labelled as 1. when -0.3, probabilities higher than
    # 0.8 are considered as 1.
    model_type = 'ml-binary'
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = 'data'

    # Load Dataset
    # Range is from 1 because the very first colomn is Chrom names.
    # Then I cut it in the preprocessing.
    cols = range(1, 20)
    # cols = [2,3,5,6,7,8,11,12,13,14,15,16,18,19,21,22,23,24,25,26,27,28,29,30,32,33,36,37,38,39,40,41,42,43,44,45,46]
    # test = preprocess.prep_test(folder, '/Test.Data.csv', cols)
    test = preprocess.prep_test(path, filename, cols)
    # _, test = preprocess.load_preprocessed_data(skip_train=True)
    scalar = StandardScaler()
    # train[:, 1:] = scalar.fit_transform(train[:, 1:])
    # joblib.dump(scalar, "mean_var")
    scaler_load = joblib.load("mean_var")
    test[:, 1:] = scaler_load.transform(test[:, 1:])

    # Build the model
    input_dim = 18
    model = nn.build_model(input_dim, nb_classes-1, type=model_type, weights_path=weights_path)
    # Print a summary of the model
    model.summary()
    score = model.evaluate(test[:, 1:], test[:, 0])
    y_pred = model.predict(test[:, 1:])
    y_test = test[:y_pred.shape[0], 0]

    y_pred_pos = np.round(y_pred + cutoff_thr)
    y_pred_neg = 1 - y_pred_pos
    y_pred_pos = np.reshape(y_pred_pos, y_pred_pos.shape[0])
    y_pred_neg = np.reshape(y_pred_neg, y_pred_neg.shape[0])

    y_pos = y_test  # np.round(y_test)
    y_neg = 1 - y_pos
    print(sum(y_neg))

    # Number of agreements between y_pos and y_pred_pos is tp
    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fn = np.sum(y_pos * y_pred_neg)
    fp = np.sum(y_neg * y_pred_pos)

    total_pos = np.sum(y_pos) + sys.float_info.epsilon
    total_neg = np.sum(y_neg) + sys.float_info.epsilon
    # print('TP: {}%, FP: {}%, TN: {}%, FN: {}%'.format(tp / total_pos, fp / total_pos, tn / total_neg, fn / total_neg))
    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp, fp, tn, fn))

    np.savetxt(folder + "/predictions.csv", y_pred, fmt='%10.5f')
    if sum(y_neg) != 0:
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
        ax2.set_ylim([thresholds[-1], thresholds[0]])
        ax2.set_xlim([fpr[0], fpr[-1]])
        plt.show()



