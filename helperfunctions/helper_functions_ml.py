from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc, RocCurveDisplay, PrecisionRecallDisplay


def computeModelMetrics(y_pred, y_pred_prob, y):
    acc = accuracy_score(y_pred, y)
    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    rocauc = roc_auc_score(y, y_pred_prob)

    return acc, f1, prec, rec, rocauc

import matplotlib.pyplot as plt

def createMetricsDF(train_acc, train_f1, train_prec, train_rec, train_rocauc,
                    dev_acc, dev_f1, dev_prec, dev_rec, dev_rocauc):
    metrics = {'Dataset Type': ['training', 'development'],
               'accuracy': [train_acc, dev_acc],
               'f1 score': [train_f1, dev_f1],
               'precision': [train_prec, dev_prec],
               'recall': [train_rec, dev_rec],
               'roc auc score': [train_rocauc, dev_rocauc]}
    metrics_df = pd.DataFrame.from_dict(metrics)

    return metrics_df

import numpy as np
import pandas as pd

def model_metric_thresh(predict_probs,
                        y_test,
                        thresh):
    preds = np.where(predict_probs >= thresh, True, False)
    return precision_score(y_test, preds, pos_label=True), recall_score(y_test, preds, pos_label=True), \
           accuracy_score(y_test, preds), f1_score(y_test, preds)

def computeAndPlotMetrics(train_pred, train_probs, dev_pred, dev_probs,
                          y_dev, y_train, thresholds, modelType):

    train_acc, train_f1, train_prec, train_rec, train_rocauc = computeModelMetrics(train_pred, train_probs, y_train)

    dev_acc, dev_f1, dev_prec, dev_rec, dev_rocauc = computeModelMetrics(dev_pred, dev_probs, y_dev)

    metrics_df = createMetricsDF(train_acc, train_f1,
                                 train_prec, train_rec, train_rocauc,
                                 dev_acc, dev_f1,
                                 dev_prec, dev_rec, dev_rocauc)

    display = RocCurveDisplay.from_predictions(y_train, train_probs)
    plt.plot(np.array([0, 1]), np.array([0, 1]))
    plt.title('Train ROC AUC Curve -  ' + modelType + ' Curve')

    display = RocCurveDisplay.from_predictions(y_dev, dev_probs)
    plt.plot(np.array([0, 1]), np.array([0, 1]))
    plt.title('Dev ROC AUC Curve -  ' + modelType + ' Curve')

    precision, recall, boundaries = precision_recall_curve(y_dev, dev_probs, pos_label=True)
    plt.figure()
    plt.plot(precision, recall)
    plt.ylabel('Recall')
    plt.xlabel('Precision')
    plt.title('Precision vs Recall from sklearn');

    # build a dataframe that stores the precision and recall for different probability thresholds
    results_df = ''
    model_metric_vals = np.array([list(model_metric_thresh(dev_probs, y_dev, i)) for i in thresholds])

    results_df = pd.DataFrame({'threshold': thresholds,
                               'precision': model_metric_vals[:, 0],
                               'recall': model_metric_vals[:, 1],
                               'accuracy': model_metric_vals[:, 2],
                               'f1': model_metric_vals[:, 3]})

    ax = results_df.plot(x='threshold', y='precision', style='ro--', label='Precision')
    results_df.plot(ax=ax, x='threshold', y='recall', style='go--', label='Recall')
    results_df.plot(ax=ax, x='threshold', y='accuracy', style='bo--', label='Accuracy')
    results_df.plot(ax=ax, x='threshold', y='f1', style='ko--', label='f1')
    plt.legend()

    return metrics_df
