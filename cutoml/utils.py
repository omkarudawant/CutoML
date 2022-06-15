"""
CutoML - A lightweight automl framework for classification and regression tasks.

Copyright (C) 2022  Omkar Udawant

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def multi_class_roc_auc_score(y_true: np.array, y_pred: np.array,
                              average='weighted'):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average=average)


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return (np.fabs(y_true - y_pred) / y_true)[mask].mean()


def classification_metrics(y_true: np.array, y_pred: np.array):
    average = 'weighted' if len(np.unique(y_true)) > 2 else None
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=average)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    roc_auc_ = multi_class_roc_auc_score(y_true=y_true, y_pred=y_pred,
                                         average=average)
    return accuracy, f1, precision, recall, roc_auc_


def regression_metrics(y_true: np.array, y_pred: np.array):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    return r2, mape, mse, mae


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "Time elapsed: {:0>2}:{:0>2}:{:05.2f} hh:mm:ss".format(
        int(hours),
        int(minutes),
        seconds)
