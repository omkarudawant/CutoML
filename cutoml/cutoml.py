"""
CutoML - A lightweight automl framework for classification and regression tasks.

Copyright (C) 2022 Omkar Udawant

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
import tqdm
from imblearn.over_sampling import SMOTE
from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from cutoml.config import Classifiers
from cutoml.config import Regressors
from cutoml.utils import timer
import time
import warnings

warnings.filterwarnings('ignore')


class CutoClassifier:
    def __init__(self, k_folds=3, oversample=False, n_jobs=1, verbose=0, random_state=0):
        self.random_state = random_state
        classifiers_config = Classifiers(
            k_folds=k_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state
        )
        self.models = classifiers_config.models
        self.oversample = oversample
        self.best_estimator = None
        self.n_jobs = n_jobs

    def _model_fitter(self, model, X, y):
        try:
            clf = Pipeline([
                ('classification_model', model)
            ])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                clf = clf.fit(X, y)
            return clf
        except Exception as e:
            pass

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        if self.oversample:
            over_sampler = SMOTE(
                random_state=self.random_state,
                k_neighbors=3,
                n_jobs=self.n_jobs
            )
            X_train, y_train = over_sampler.fit_resample(
                X_train, y_train
            )

        start_time = time.time()

        pool = ProcessingPool()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                *trained_pipelines, = tqdm.tqdm(
                    pool.imap(
                        lambda x: self._model_fitter(x,
                                                     X_train,
                                                     y_train),
                        self.models,
                        chunksize=len(self.models) // 2
                    ),
                    total=len(self.models),
                    desc='Optimizing'
                )

                end_time = time.time()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            finally:
                pool.close()
                pool.join()
        print(timer(start=start_time, end=end_time))
        trained_models = dict()
        for pipeline in trained_pipelines:
            if pipeline:
                try:
                    score = pipeline.score(X_test, y_test)
                    trained_models[score] = pipeline
                except Exception as e:
                    pass
        if trained_models:
            self.best_estimator = max(
                sorted(trained_models.items(), reverse=True))[1]
            return self
        else:
            raise RuntimeError('Could not find best estimator.')

    def predict(self, X):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call fit() method first.')
        prediction = self.best_estimator.predict(X)
        return prediction

    def predict_proba(self, X):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call fit() method first.')
        prediction_probablity = self.best_estimator.predict_proba(X)
        return prediction_probablity

    def score(self, X, y):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call fit() method first.')
        score = self.best_estimator.score(X, y)
        return score


class CutoRegressor:
    def __init__(self, k_folds=3, n_jobs=1, verbose=0, random_state=0):
        self.random_state = random_state
        regressors_config = Regressors(
            k_folds=k_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state
        )
        self.models = regressors_config.models
        self.best_estimator = None
        self.n_jobs = n_jobs

    def _model_fitter(self, model, X, y):
        try:
            rgr = Pipeline([
                ('regression_model', model)
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rgr = rgr.fit(X, y)
            return rgr
        except Exception as e:
            print(e)
            pass

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        start_time = time.time()
        pool = ProcessingPool()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                *trained_pipelines, = tqdm.tqdm(
                    pool.imap(
                        lambda x: self._model_fitter(x,
                                                     X_train,
                                                     y_train),
                        self.models,
                        chunksize=len(self.models) // 2
                    ),
                    total=len(self.models),
                    desc='Optimizing'
                )

                end_time = time.time()
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
            finally:
                pool.close()
                pool.join()
        print(timer(start=start_time, end=end_time))
        trained_models = dict()
        for pipeline in trained_pipelines:
            if pipeline:
                try:
                    score = pipeline.score(X_test, y_test)
                    trained_models[score] = pipeline
                except Exception as e:
                    pass
        if trained_models:
            self.best_estimator = max(
                sorted(trained_models.items(), reverse=True))[1]
            return self
        else:
            raise RuntimeError('Could not find best estimator.')

    def predict(self, X):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call fit() method first.')
        prediction = self.best_estimator.predict(X)
        return prediction

    def score(self, X, y):
        if not self.best_estimator:
            raise RuntimeError(
                'Models not fit yet, please call fit() method first.')
        score = self.best_estimator.score(X, y)
        return score
