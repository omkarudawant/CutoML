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

# Classifiers
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

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import numpy as np


class Classifiers:
    def __init__(self,
                 k_folds=5,
                 n_jobs=2,
                 verbose=1,
                 random_state=0):
        self.models = [
            HalvingRandomSearchCV(
                estimator=GaussianNB(),
                param_distributions={
                    'var_smoothing': np.logspace(0, -9, 10)
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='f1_weighted',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=BernoulliNB(),
                param_distributions={
                    'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 20.],
                    'fit_prior': [True, False]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='f1_weighted',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=MultinomialNB(),
                param_distributions={
                    'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 20.],
                    'fit_prior': [True, False]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='f1_weighted',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                ExtraTreesClassifier(n_jobs=n_jobs,
                                     random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'criterion': ["gini", "entropy"],
                    'max_features': np.arange(0.05, 1.00, 0.05),
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'bootstrap': [True, False]
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                KNeighborsClassifier(n_jobs=n_jobs),
                param_distributions={
                    'n_neighbors': [2, 4, 8, 10, 12],
                    'weights': ["uniform", "distance"],
                    'p': [1, 2]
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                LogisticRegression(n_jobs=n_jobs,
                                   random_state=random_state),
                param_distributions={
                    'penalty': ["l1", "l2"],
                    'C': np.logspace(0, 3, 4),
                    'max_iter': [100, 500]
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                SGDClassifier(n_jobs=n_jobs,
                              random_state=random_state),
                param_distributions={
                    'loss': ['log', 'hinge', 'modified_huber', 'squared_hinge',
                             'perceptron'],
                    'penalty': ['elasticnet'],
                    'alpha': [0.0, 0.01, 0.001],
                    'learning_rate': ['invscaling', 'constant'],
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                LinearSVC(random_state=random_state),
                param_distributions={
                    'penalty': ["l2"],
                    'dual': [True],
                    'max_iter': [100, 500],
                    'tol': [1e-3, 1e-2, 1e-1],
                    'C': np.logspace(0, 3, 4),
                },
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                cv=k_folds,
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                LinearSVC(random_state=random_state),
                param_distributions={
                    'penalty': ["l1", "l2"],
                    'max_iter': [100, 500],
                    'dual': [False],
                    'tol': [1e-3, 1e-2, 1e-1],
                    'C': np.logspace(0, 3, 4)
                },
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                cv=k_folds,
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                RandomForestClassifier(
                    n_jobs=n_jobs,
                    random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'max_features': ['log2', 'sqrt'],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                GradientBoostingClassifier(random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'learning_rate': [1e-3, 1e-2, 1e-1, 1.],
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                XGBClassifier(n_jobs=n_jobs,
                              random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'learning_rate': [1e-3, 1e-2, 1e-1, 1.],
                    'min_child_weight': range(1, 21),
                    'subsample': np.arange(0.05, 1.0, 0.05),
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                DecisionTreeClassifier(random_state=random_state),
                param_distributions={
                    'criterion': ["gini", "entropy"],
                    'max_depth': range(1, 21),
                    'min_samples_split': range(2, 21),
                    'min_samples_leaf': range(1, 21)
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                scoring='f1_weighted',
                n_jobs=n_jobs
            ),
            HalvingRandomSearchCV(
                MLPClassifier(random_state=random_state),
                param_distributions={
                    'max_iter': [100, 500],
                    'hidden_layer_sizes': [
                        (10, 30, 10),
                        (50, 50, 50),
                        (50, 100, 50),
                        (20,),
                        (100,)
                    ],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant', 'adaptive'],
                },
                cv=k_folds,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='f1_weighted',
            )
        ]


class Regressors:
    def __init__(self,
                 k_folds=5,
                 n_jobs=2,
                 verbose=1,
                 random_state=0):
        self.models = [
            LinearRegression(n_jobs=n_jobs),
            HalvingRandomSearchCV(
                estimator=ElasticNet(random_state=random_state),
                param_distributions={
                    'l1_ratio': np.arange(0.0, 1.0, 0.05),
                    'tol': [1e-3, 1e-2, 1e-1]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=ExtraTreesRegressor(random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'criterion': ["gini", "entropy"],
                    'max_features': np.arange(0.05, 1.00, 0.05),
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'bootstrap': [True, False]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=GradientBoostingRegressor(random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'loss': ["ls", "lad", "huber", "quantile"],
                    'learning_rate': [1e-3, 1e-2, 1e-1, 1.],
                    'max_features': ['log2', 'sqrt'],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=AdaBoostRegressor(random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'learning_rate': [1e-3, 1e-2, 1e-1, 1.],
                    'loss': ["linear", "square", "exponential"]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=DecisionTreeRegressor(random_state=random_state),
                param_distributions={
                    'max_depth': range(1, 11),
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=KNeighborsRegressor(n_jobs=n_jobs),
                param_distributions={
                    'n_neighbors': [2, 4, 8, 10, 12],
                    'p': [1, 2]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=Ridge(random_state=random_state),
                param_distributions={
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=Lasso(random_state=random_state),
                param_distributions={
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=LassoLars(random_state=random_state),
                param_distributions={
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=LinearSVR(random_state=random_state),
                param_distributions={
                    'loss': ["epsilon_insensitive",
                             "squared_epsilon_insensitive"],
                    'dual': [True],
                    'tol': [1e-3, 1e-2, 1e-1],
                    'C': np.logspace(0, 3, 4),
                    'epsilon': [1e-3, 1e-2, 1e-1, 1.]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=LinearSVR(random_state=random_state),
                param_distributions={
                    'loss': ["squared_epsilon_insensitive"],
                    'dual': [False],
                    'tol': [1e-3, 1e-2, 1e-1],
                    'C': np.logspace(0, 3, 4),
                    'epsilon': [1e-3, 1e-2, 1e-1, 1.]
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=RandomForestRegressor(
                    n_jobs=n_jobs,
                    random_state=random_state),
                param_distributions={
                    'max_iter': [100],
                    'max_features': ['log2', 'sqrt', ],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=XGBRegressor(
                    n_jobs=n_jobs,
                    random_state=random_state),
                param_distributions={
                    'n_estimators': [100],
                    'learning_rate': [1e-3, 1e-2, 1e-1, 1.],
                    'min_child_weight': range(1, 21),
                    'subsample': np.arange(0.05, 1.0, 0.05),
                    'min_child_weight': range(1, 21),
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            ),
            HalvingRandomSearchCV(
                estimator=SGDRegressor(random_state=random_state),
                param_distributions={
                    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
                    'penalty': ['elasticnet'],
                    'alpha': [0.0, 0.01, 0.001],
                    'learning_rate': ['invscaling', 'constant'],
                },
                cv=k_folds,
                verbose=verbose,
                n_jobs=n_jobs,
                scoring='r2',
                random_state=random_state
            )
        ]
