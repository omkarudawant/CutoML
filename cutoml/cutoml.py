"""
CutoML - A lightweight automl framework for classification and regression tasks.

Copyright (C) 2021 Omkar Udawant

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

from cutoml.utils import regression_metrics
from cutoml.utils import classification_metrics
from cutoml.utils import timer
from cutoml.config import Classifiers
from cutoml.config import Regressors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing
import numpy as np
import time
import json
import tqdm


class CutoClassifier:
    def __init__(self, k_folds=3, n_jobs=multiprocessing.cpu_count() // 2, verbose=0):
        self.models = Classifiers(
            k_folds=k_folds, n_jobs=n_jobs, verbose=verbose)
        self.models = self.models.models
        self.best_estimator = None
        self.n_jobs = n_jobs

    def _model_fitter(sef, model, X, y):
        try:
            clf = Pipeline([
                ('classification_model', model)
            ])
            clf = clf.fit(X, y)
            return clf
        except Exception as e:
            print(
                f'Skipping {clf.named_steps["classification_model"]} because of {e}')

    def fit(self, X, y):
        assert len(X) > len(np.unique(y)), (
            "Features available for " "number of classes are not enough"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        start_time = time.time()
        pool = Pool(nodes=self.n_jobs)
        try:
            *trained_pipelines, = tqdm.tqdm(pool.map(lambda x: self._model_fitter(x,
                                                                                  X_train,
                                                                                  y_train),
                                                     tqdm.tqdm(desc="Optimizing Classifiers",
                                                               iterable=self.models,
                                                               total=len(
                                                                   self.models),
                                                               )
                                                     )
                                            )
            end_time = time.time()
        finally:
            pool.close()
        print(timer(start=start_time, end=end_time))

        trained_models = dict()
        for pipeline in trained_pipelines:
            if pipeline:
                try:
                    pred = pipeline.predict(X_test)
                    acc, f1, prec, recall, roc_auc = classification_metrics(
                        y_true=y_test, y_pred=pred
                    )
                    trained_models[f1] = pipeline
                except Exception as e:
                    print(
                        f'\nSkipping {pipeline.named_steps["classification_model"].best_estimator_} because of {e}')

        if trained_models:
            self.best_estimator = max(
                sorted(trained_models.items(), reverse=True))[1]
            return self
        else:
            raise RuntimeError('Could not find best estimator.')

    def predict(self, X):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call object.fit() method first.')
        prediction = self.best_estimator.predict(X)
        return prediction

    def predict_proba(self, X):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call object.fit() method first.')
        prediction_probablity = self.best_estimator.predict_proba(X)
        return prediction_probablity

    def score(self, X, y):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call object.fit() method first.')
        pred = self.best_estimator.predict(X)
        accuracy, f1, precision, recall, roc_auc_ = classification_metrics(
            y_true=y, y_pred=pred
        )
        scores = {
            "Accuracy": accuracy,
            "F1 score": f1,
            "Precision": precision,
            "Recall": recall,
            "ROC_AUC_score": roc_auc_,
        }
        return json.dumps(scores, sort_keys=True)


class CutoRegressor:
    def __init__(self, k_folds=3, n_jobs=multiprocessing.cpu_count() // 2, verbose=0):
        self.models = Regressors(
            k_folds=k_folds, n_jobs=n_jobs, verbose=verbose)
        self.models = self.models.models
        self.best_estimator = None
        self.n_jobs = n_jobs

    def _model_fitter(sef, model, X, y):
        try:
            rgr = Pipeline([
                ('regression_model', model)
            ])
            rgr = rgr.fit(X, y)
            return rgr
        except Exception as e:
            print(
                f'Skipping {rgr.named_steps["regression_model"]} because of {e}')

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        start_time = time.time()
        pool = Pool(nodes=self.n_jobs)
        try:
            *trained_pipelines, = tqdm.tqdm(pool.map(lambda x: self._model_fitter(x,
                                                                                  X_train,
                                                                                  y_train),
                                                     tqdm.tqdm(desc="Optimizing Regressors",
                                                               iterable=self.models,
                                                               total=len(
                                                                   self.models),
                                                               )
                                                     )
                                            )
            end_time = time.time()
        finally:
            pool.close()
        print(timer(start=start_time, end=end_time))

        trained_models = dict()
        for pipeline in trained_pipelines:
            if pipeline:
                try:
                    pred = pipeline.predict(X_test)
                    r2, mape, mse, mae = regression_metrics(y_true=y_test,
                                                            y_pred=pred)
                    trained_models[r2] = pipeline
                except Exception as e:
                    print(
                        f'\nSkipping {pipeline.named_steps["regression_model"].best_estimator_} because of {e}')
        if trained_models:
            self.best_estimator = max(
                sorted(trained_models.items(), reverse=True))[1]
            return self
        else:
            raise RuntimeError('Could not find best estimator.')

    def predict(self, X):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call object.fit() method first.')
        prediction = self.best_estimator.predict(X)
        return prediction

    def score(self, X, y):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call object.fit() method first.')
        pred = self.best_estimator.predict(X)
        r2, mape, mse, mae = regression_metrics(y_true=y, y_pred=pred)
        scores = {"R2 score": r2, "MAPE": mape, "MSE": mse, "MAE": mae}
        return json.dumps(scores, sort_keys=True)
