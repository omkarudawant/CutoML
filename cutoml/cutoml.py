"""
CutaML - A lightweight automl framework for classification and regression tasks.

Copyright (C) 2021  Omkar Udawant

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
from warnings import filterwarnings

filterwarnings('ignore')

from cutoml.config import classifiers
from cutoml.config import regressors
from cutoml.utils import timer
from cutoml.utils import classification_metrics, regression_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import time
import json


class CutoClassifier:
    def __init__(self):
        self.models = classifiers.models
        self.best_estimator = None

    def fit(self, X, y):
        assert len(X) > len(np.unique(y)), "Features available for " \
                                           "number of classes are not enough"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.1,
            random_state=0
        )
        trained_models = dict()
        start_time = time.time()
        for model in self.models:
            try:
                clf = Pipeline([
                    ('standard_scale', StandardScaler()),
                    ('classification_model', model)
                ])
                start = time.time()
                clf.fit(X=X_train, y=y_train)
                end = time.time()
                timer(start=start, end=end)
                pred = clf.predict(X_test)

                acc, f1, prec, recall, roc_auc = classification_metrics(
                    y_true=y_test,
                    y_pred=pred
                )

                trained_models[f1] = clf
            except Exception as e:
                print(e)
        end_time = time.time()
        print(timer(start=start_time, end=end_time))
        self.best_estimator = max(
            sorted(trained_models.items(), reverse=True))[1]

    def predict(self, X):
        prediction = self.best_estimator.predict(X)
        return prediction

    def predict_proba(self, X):
        prediction_probablity = self.best_estimator.predict_proba(X)
        return prediction_probablity

    def score(self, X, y):
        assert self.best_estimator, "Models not fitted yet"
        pred = self.best_estimator.predict(X)
        accuracy, f1, precision, recall, roc_auc_ = classification_metrics(
            y_true=y,
            y_pred=pred
        )

        scores = {
            'Accuracy': accuracy,
            'F1 score': f1,
            'Precision': precision,
            'Recall': recall,
            'ROC_AUC_score': roc_auc_
        }
        return json.dumps(scores, indent=4, sort_keys=True)


class CutoRegressor:
    def __init__(self):
        self.models = regressors.models
        self.best_estimator = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.1,
            random_state=0
        )
        trained_models = dict()
        start_time = time.time()
        for model in self.models:
            clf = Pipeline([
                ('standard_scale', StandardScaler()),
                ('regression_model', model)
            ])
            start = time.time()
            clf.fit(X=X_train, y=y_train)
            end = time.time()
            timer(start=start, end=end)

            pred = clf.predict(X_test)
            r2, mape, mse, mae = regression_metrics(
                y_true=y_test,
                y_pred=pred
            )
            trained_models[r2] = clf
        end_time = time.time()
        timer(start=start_time, end=end_time)
        self.best_estimator = max(
            sorted(trained_models.items(), reverse=True))[1]

    def predict(self, X):
        prediction = self.best_estimator.predict(X)
        return prediction

    def score(self, X, y):
        assert self.best_estimator, "Models not fitted yet"
        pred = self.best_estimator.predict(X)
        r2, mape, mse, mae = regression_metrics(
            y_true=y,
            y_pred=pred
        )
        scores = {
            'R2 score': r2,
            'MAPE': mape,
            'MSE': mse,
            'MAE': mae
        }
        return json.dumps(scores, indent=2, sort_keys=True)
