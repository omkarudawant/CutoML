from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score

import numpy as np


def classification_metrics(y_true: np.array, y_pred: np.array):
    # TODO: Add ROC_AUC score for multiclass classification
    assert len(np.unique(y_true)) == len(np.unique(y_pred)), "Number of " \
                                                             "unique classes " \
                                                             "in y_true and " \
                                                             "y_pred not " \
                                                             "matching"
    average = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

    print(f" Accuracy: {accuracy} | F1: {f1} | Precision: "
          f"{precision} | Recall: {recall}")

    return accuracy, f1, precision, recall


def regression_metrics(y_true: np.array, y_pred: np.array):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    print(f" R2 score: {r2} | MSE: {mse} | MAE: "
          f"{mae} | MAPE: {mape}")
    return r2, mape, mse, mae


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        "Time elapsed: {:0>2}:{:0>2}:{:05.2f} hh:mm:ss".format(
            int(hours),
            int(minutes),
            seconds)
    )
