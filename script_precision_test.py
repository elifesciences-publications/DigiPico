import numpy as np

def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)

    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp, fp, tn, fn))

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + np.finfo(np.float32).eps)


def mcor_fixed(y_true, y_pred):
    # matthews_correlation
    y_pred = np.round(np.clip(y_pred, 0, 1))

    tp = np.sum(y_true[:, 1] * y_pred[:, 1])
    tn = np.sum(y_true[:, 0] * y_pred[:, 0])

    total_pos = np.sum(y_true[:, 1])
    total_neg = np.sum(y_true[:, 0])

    fp = total_pos - tp
    fn = total_neg - tn

    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(tp, fp, tn, fn))

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + np.finfo(np.float32).eps)

print("Results 1:")
y_true = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
y_pred = y_true
print("Original mcor:")
print(mcor(y_true,y_pred))
print("Fixed mcor:")
print(mcor_fixed(y_true,y_pred))

print("Results 2:")
y_true = np.array([[0., 1.], [0., 1.], [1., 0.], [1., 0.]])
y_pred = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]])
print("Original mcor:")
print(mcor(y_true,y_pred))
print("Fixed mcor:")
print(mcor_fixed(y_true,y_pred))