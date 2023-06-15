import numpy as np


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        tp =tn= fp= fn = 0
        for i in range(len(actual)):
            if actual[i] == predictions[i]:
                if actual[i] == 1:
                    tp += 1
                else:
                    tn += 1
            elif actual[i] == 1:
                fn += 1
            else:
                fp += 1

        confusion = np.array([[ tn, fp], [ fn, tp]])
        return confusion

    raise NotImplementedError


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        con_mat = compute_confusion_matrix(actual, predictions)
        correct = con_mat[0,0] + con_mat[1,1]
        total = con_mat[0,0] + con_mat[0,1] + con_mat[1,0]+ con_mat[1,1]
        return correct / total
    raise NotImplementedError


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        con_mat = compute_confusion_matrix(actual, predictions)
        tp = con_mat[1,1]
        fp = con_mat[0,1]
        fn = con_mat[1,0]
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = np.nan
        try:
            r = tp / (tp + fn)
        except ZeroDivisionError:
            r = np.nan
        return (p, r)
    raise NotImplementedError


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    else:
        p, r = compute_precision_and_recall(actual, predictions)
        if p == np.nan or r == np.nan or p == 0:
            return np.nan
        else: 
            return (2 * p * r)/(p + r)
    raise NotImplementedError
