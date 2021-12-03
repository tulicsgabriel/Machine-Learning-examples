# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:04:22 2021

@author: MIKLOS
"""

import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from minepy import MINE
from scipy.spatial import distance
import scipy.stats as stats

# generate random floating point values
from numpy.random import seed
from numpy.random import rand

# seed random number generator
seed(1)


def get_accuracy(mx):
    """ Gets accuracy
        (tp + tn) / (tp + fp + tn + fn)
    """
    [tp, fp], [fn, tn] = mx
    return (tp + tn) / (tp + fp + tn + fn)


def get_recall(mx):
    """ Gets sensitivity, recall, hit rate, or true positive rate (TPR)
        Same as sensitivity
    """
    [tp, fp], [fn, tn] = mx
    return tp / (tp + fn)


def get_precision(mx):
    """ Gets precision or positive predictive value (PPV)
    """
    [tp, fp], [fn, tn] = mx
    return tp / (tp + fp)


def get_f1score(mx):
    """ Gets F1 score, the harmonic mean of precision and sensitivity
        2*((precision*recall)/(precision+recall))
    """
    return 2 * (
        (get_precision(mx) * get_recall(mx)) / (get_precision(mx) + get_recall(mx))
    )


def get_specificity(mx):
    """ Gets specificity, selectivity or true negative rate (TNR)
    """
    [tp, fp], [fn, tn] = mx
    return tn / (tn + fp)


def get_sensitivity(mx):
    """ Gets sensitivity
    """
    [tp, fp], [fn, tn] = mx
    return tp / (tp + fn)


def get_MCC(mx):
    """" Gets the Matthews Correlation Coefficient (MCC)
    """
    [tp, fp], [fn, tn] = mx
    return (tp * tn - fp * fn) / math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )


def print_confusion_matrix(y_true, y_pred):
    """ Print confusion matrix and return confusion matrix as array of ints
    """
    print("")

    true = pd.Categorical(
        list(np.where(np.array(y_true) == 1, "Cancer", "Healthy")),
        categories=["Cancer", "Healthy"],
    )

    pred = pd.Categorical(
        list(np.where(np.array(y_pred) == 1, "Cancer", "Healthy")),
        categories=["Cancer", "Healthy"],
    )

    df_finalconf = pd.crosstab(
        pred,
        true,
        rownames=["Predicted"],
        colnames=["Actual"],
        margins=True,
        margins_name="Total",
    )

    print(df_finalconf)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm_final = np.array([[tp, fp], [fn, tn]])

    return cm_final


def get_confusion_matrix(y_true, y_pred):
    """ returns confusion matrix as array of ints
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm_final = np.array([[tp, fp], [fn, tn]])

    return cm_final


def get_number_of_proteins(feature_list):
    """ Input a list of features and outputs the number of proteins present
    in the feature vector
    """
    protein_names = (
        "prot1",
        "prot2",
        "prot3",
        "prot4",
        "prot5",
        "prot6",
        "prot7",
        "prot6",
        "prot7",
        "prot8",
        "prot9",
        "prot10"
    )
    bool_list = ["False"] * len(protein_names)

    # make features upper in feature list
    feature_list = [w.upper() for w in feature_list]

    for i, protein in enumerate(protein_names):
        for feature in feature_list:
            if protein in feature:
                bool_list[i] = "True"

    num_of_proteins = bool_list.count("True")
    return num_of_proteins


def get_correlation_metrics(x, y):
    """
    Get of MIC, Pearson, Spearman and Cosine similarity
    More on MIC: https://rhondenewint.wordpress.com/2018/12/09/maximal-information-coefficient-a-modern-approach-for-finding-associations-in-large-data-sets/

    Parameters
    ----------
    def get_correlation_metrics : x, y lists of numerical values.

    Returns
    -------
    Series of Pearson coeff, Spearman coeff, MIC & Cosine Similarity.

    """
    # initialise mine alogithm with default parameters
    mine = MINE(alpha=0.6, c=15)

    results = dict()

    # returns both Pearson's coefficient and p-value,
    # keep the first value which is the r coefficient
    results["Pearson coeff"] = stats.pearsonr(x, y)[0]
    results["Spearman coeff"] = stats.spearmanr(x, y)[0]
    mine.compute_score(x, y)
    results["MIC"] = mine.mic()
    results["Cosine Similarity"] = 1 - distance.cosine(x, y)
    return pd.Series(results)


if __name__ == "__main__":
    y = [1, 0, 1, 1, 0, 1]
    pred = [0, 0, 1, 0, 0, 1]

    print_confusion_matrix(y, pred)
    mx_ = get_confusion_matrix(y, pred)

    print("\n")
    print("Mcc: ", get_MCC(mx_))
    print("Acc.: ", get_accuracy(mx_))
    print("F1 score: ", get_f1score(mx_))
    print("Precision: ", get_precision(mx_))
    print("Recall: ", get_recall(mx_))
    print("Sen.: ", get_sensitivity(mx_))
    print("Spec.:", get_sensitivity(mx_))
    print("\n")

    manual_feature_names = (
        "(prot9 - exp(prot10))**3",
        "(-prot10**3 + Abs(prot4))**2",
        "(-prot10 + Abs(prot3))**3",
        "(prot1 + Edad_scaled**3)**3",
        "(prot9 + Abs(prot6))**3",
        "prot9*exp(-prot6)",
        "(prot9 + Abs(prot3))**2",
        "(-prot2**3 + prot10**2)**3",
    )

    print("Number of proteins in the feature vector: ")
    print(get_number_of_proteins(manual_feature_names))
    print("\n")
    ##
    # generate random numbers between 0-1 *10
    x = list((rand(10) * 10))
    y = [2.0 + 0.7 * num ** 2 + 0.5 * num for num in x]

    print("---------------------------")
    out = get_correlation_metrics(x, y)
    print(out)
