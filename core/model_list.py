from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
import multiprocessing
import numpy as np

cross_val = 3
scorer = make_scorer(roc_auc_score, average='weighted', multi_class='ovr')
models = [
    MultinomialNB(),
    RandomizedSearchCV(
        LogisticRegression(n_jobs=multiprocessing.cpu_count() // 2,
                           random_state=0),
        param_distributions={
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1, 10, 100],
            'max_iter': [100, 500, 1000]
        },
        cv=cross_val,
        random_state=0,
        verbose=2
    ),
    RandomizedSearchCV(
        SGDClassifier(n_jobs=multiprocessing.cpu_count() // 2,
                      random_state=0),
        param_distributions={
            'penalty': ['l1', 'l2'],
            'alpha': [0.1, 1, 10, 100, 1000]
        },
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        LinearSVC(random_state=0),
        param_distributions={
            'C': [0.1, 1, 10, 100, 1000]},
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
            'n_estimators': np.arange(start=100,
                                      stop=1100,
                                      step=400),
            'max_features': ['auto', 'sqrt', 'log2']},
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2,
                      random_state=0),
        param_distributions={
            'n_estimators': np.arange(start=100,
                                      stop=1100,
                                      step=400),
            'max_depth': np.arange(start=3,
                                   stop=13,
                                   step=2)},
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    ),
    RandomizedSearchCV(
        DecisionTreeClassifier(random_state=0),
        param_distributions={
            'criterion': ['gini', 'entropy'],
            'max_depth': np.arange(start=2,
                                   stop=7,
                                   step=2)},
        cv=cross_val,
        random_state=0,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2
    )
]
