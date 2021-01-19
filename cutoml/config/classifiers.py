"""
CutoML - A lightweight automl framework for classification and regression tasks.

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

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

import multiprocessing
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class Classifiers:
    def __init__(self,
                 k_folds=5,
                 n_jobs=multiprocessing.cpu_count() // 2,
                 verbose=1):
        self.models = [
            GaussianNB(),
            RandomizedSearchCV(
                estimator=BernoulliNB(),
                param_distributions={
                    'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                    'fit_prior': [True, False]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                random_state=0
            ),
            RandomizedSearchCV(
                estimator=MultinomialNB(),
                param_distributions={
                    'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
                    'fit_prior': [True, False]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                random_state=0
            ),
            RandomizedSearchCV(
                ExtraTreesClassifier(n_jobs=n_jobs,
                                     random_state=0),
                param_distributions={
                    'n_estimators': np.arange(start=100, stop=500, step=100),
                    'criterion': ["gini", "entropy"],
                    'max_features': np.arange(0.05, 1.01, 0.05),
                    'min_samples_split': range(2, 21),
                    'min_samples_leaf': range(1, 21),
                    'bootstrap': [True, False]
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                KNeighborsClassifier(n_jobs=n_jobs),
                param_distributions={
                    'n_neighbors': range(1, 101),
                    'weights': ["uniform", "distance"],
                    'p': [1, 2]
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                LogisticRegression(n_jobs=n_jobs,
                                   random_state=0),
                param_distributions={
                    'penalty': ["l1", "l2"],
                    'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20.,
                          25.],
                    'dual': [True, False],
                    'max_iter': np.arange(start=100, stop=500, step=100)
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                SGDClassifier(n_jobs=n_jobs,
                              random_state=0),
                param_distributions={
                    'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge',
                             'perceptron'],
                    'penalty': ['elasticnet'],
                    'alpha': [0.0, 0.01, 0.001],
                    'learning_rate': ['invscaling', 'constant'],
                    'fit_intercept': [True, False],
                    'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
                    'eta0': [0.1, 1.0, 0.01],
                    'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                LinearSVC(random_state=0),
                param_distributions={
                    'penalty': ["l1", "l2"],
                    'loss': ["hinge", "squared_hinge"],
                    'dual': [True, False],
                    'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                    'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20.,
                          25.]
                },
                random_state=0,
                verbose=verbose,
                cv=k_folds,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                RandomForestClassifier(
                    n_jobs=n_jobs,
                    random_state=0),
                param_distributions={
                    'n_estimators': np.arange(start=100, stop=500, step=100),
                    'criterion': ["gini", "entropy"],
                    'max_features': np.arange(0.05, 1.01, 0.05),
                    'min_samples_split': range(2, 21),
                    'min_samples_leaf': range(1, 21),
                    'bootstrap': [True, False]
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                GradientBoostingClassifier(random_state=0),
                param_distributions={
                    'n_estimators': np.arange(start=100, stop=500, step=100),
                    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                    'max_depth': range(1, 11),
                    'min_samples_split': range(2, 21),
                    'min_samples_leaf': range(1, 21),
                    'subsample': np.arange(0.05, 1.01, 0.05),
                    'max_features': np.arange(0.05, 1.01, 0.05)
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                XGBClassifier(n_jobs=n_jobs,
                              random_state=0),
                param_distributions={
                    'n_estimators': np.arange(start=100, stop=500, step=100),
                    'max_depth': range(1, 11),
                    'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                    'subsample': np.arange(0.05, 1.01, 0.05),
                    'min_child_weight': range(1, 21),
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                DecisionTreeClassifier(random_state=0),
                param_distributions={
                    'criterion': ["gini", "entropy"],
                    'max_depth': range(1, 11),
                    'min_samples_split': range(2, 21),
                    'min_samples_leaf': range(1, 21)
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            ),
            RandomizedSearchCV(
                MLPClassifier(random_state=0),
                param_distributions={
                    'max_iter': np.arange(start=100, stop=1000, step=100),
                    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
                    'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
                },
                cv=k_folds,
                random_state=0,
                verbose=verbose,
                n_jobs=n_jobs
            )
        ]
