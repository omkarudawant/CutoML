from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import RandomizedSearchCV

import multiprocessing
import numpy as np

cross_val = 10
models = [
    GaussianNB(),
    RandomizedSearchCV(
        estimator=BernoulliNB(),
        param_distributions={
            'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=MultinomialNB(),
        param_distributions={
            'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
            'fit_prior': [True, False]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        ExtraTreesClassifier(n_jobs=multiprocessing.cpu_count() // 2,
                             random_state=0),
        param_distributions={
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        KNeighborsClassifier(n_jobs=multiprocessing.cpu_count() // 2),
        param_distributions={
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        LogisticRegression(n_jobs=multiprocessing.cpu_count() // 2,
                           random_state=0),
        param_distributions={
            'penalty': ["l1", "l2"],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'dual': [True, False],
            'max_iter': [100, 200]
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        SGDClassifier(n_jobs=multiprocessing.cpu_count() // 2,
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
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        LinearSVC(random_state=0),
        param_distributions={
            'penalty': ["l1", "l2"],
            'loss': ["hinge", "squared_hinge"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
        },
        random_state=0,
        verbose=2,
        cv=cross_val,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        RandomForestClassifier(
            n_jobs=multiprocessing.cpu_count() // 2,
            random_state=0),
        param_distributions={
            'n_estimators': [100],
            'criterion': ["gini", "entropy"],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        GradientBoostingClassifier(random_state=0),
        param_distributions={
            'n_estimators': [100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05)
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2,
                      random_state=0),
        param_distributions={
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        DecisionTreeClassifier(random_state=0),
        param_distributions={
            'criterion': ["gini", "entropy"],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        MLPClassifier(random_state=0),
        param_distributions={
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
            'learning_rate_init': [1e-3, 1e-2, 1e-1, 0.5, 1.]
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    )
]
