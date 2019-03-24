from __future__ import print_function
import numpy as np
import nn
import preprocess
import keras
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.externals import joblib

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Not enough arguments. Please enter sample name, train path, out path, threshold, and number of columns")
    sample = sys.argv[1]
    train_pth = sys.argv[2]
    out_pth = sys.argv[3]
    print("Threshold: ", sys.argv[4])
    cutoff_thr = 0.5 - float(sys.argv[4])
    print("col numbers: ", float(sys.argv[5]))

    weights_path = train_pth + '/train/mutation_logistic_wts.h5'
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    nb_classes = 2
    model_type = 'ml-binary'
    folder = out_pth + "/"

    # Load Dataset
    # Range is from 1 because the very first colomn is Chrom names.
    # Then I cut it in the preprocessing.
    cols = range(1, int(sys.argv[5]))
    print("col numbers: ", cols)
    test = preprocess.prep_test(folder, sample + '.data.csv', cols)
    print(np.shape(test))
    scaler = joblib.load(train_pth + '/train/mean_var')
    test[:, 1:] = scaler.transform(test[:, 1:])

    # Build the model
    input_dim = int(sys.argv[5]) - 2
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

    np.savetxt(folder + "predictions.csv", y_pred, fmt='%10.5f')
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
        plt.savefig(folder + sample + ".ROC.apply.png")



