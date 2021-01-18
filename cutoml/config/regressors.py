from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, \
    LassoLars, SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, \
    RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV
import multiprocessing
import numpy as np

cross_val = 10
models = [
    LinearRegression(n_jobs=multiprocessing.cpu_count() // 2),
    RandomizedSearchCV(
        estimator=ElasticNet(random_state=0),
        param_distributions={
            'l1_ratio': np.arange(0.0, 1.01, 0.05),
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=ExtraTreesRegressor(random_state=0),
        param_distributions={
            'n_estimators': [100],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=0),
        param_distributions={
            'n_estimators': [100],
            'loss': ["ls", "lad", "huber", "quantile"],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05),
            'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=AdaBoostRegressor(random_state=0),
        param_distributions={
            'n_estimators': [100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'loss': ["linear", "square", "exponential"]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=DecisionTreeRegressor(random_state=0),
        param_distributions={
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21)
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=KNeighborsRegressor(n_jobs=multiprocessing.cpu_count() // 2),
        param_distributions={
            'n_neighbors': range(1, 101),
            'weights': ["uniform", "distance"],
            'p': [1, 2]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=Ridge(random_state=0),
        param_distributions={
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=Lasso(random_state=0),
        param_distributions={
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=LassoLars(random_state=0),
        param_distributions={
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=LinearSVR(random_state=0),
        param_distributions={
            'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
            'dual': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
            'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=RandomForestRegressor(
            n_jobs=multiprocessing.cpu_count() // 2,
            random_state=0),
        param_distributions={
            'n_estimators': [100],
            'max_features': np.arange(0.05, 1.01, 0.05),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'bootstrap': [True, False]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=XGBRegressor(
            n_jobs=multiprocessing.cpu_count() // 2,
            random_state=0),
        param_distributions={
            'n_estimators': [100],
            'max_depth': range(1, 11),
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'subsample': np.arange(0.05, 1.01, 0.05),
            'min_child_weight': range(1, 21),
            'objective': ['reg:squarederror']
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    ),
    RandomizedSearchCV(
        estimator=SGDRegressor(random_state=0),
        param_distributions={
            'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
            'penalty': ['elasticnet'],
            'alpha': [0.0, 0.01, 0.001],
            'learning_rate': ['invscaling', 'constant'],
            'fit_intercept': [True, False],
            'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
            'eta0': [0.1, 1.0, 0.01],
            'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
        },
        cv=cross_val,
        verbose=2,
        n_jobs=multiprocessing.cpu_count() // 2,
        random_state=0
    )
]
