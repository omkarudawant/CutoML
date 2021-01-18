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
    super(CutoClassifier, self).__init__()

    def __init__(self, test_size=0.2):
        self.models = classifiers.models
        self.best_estimator = None
        self.test_size = test_size

    def fit(self, X, y):
        assert len(X) > len(np.unique(y)), "Features available for " \
                                           "number of classes are not enough"
        X_train, y_train, X_test, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=0
        )
        trained_models = dict()
        for model in self.models:
            clf = Pipeline([
                ('standard_scale', StandardScaler()),
                ('classification_model', model)
            ])
            start = time.time()
            clf.fit(X=X, y=y)
            end = time.time()
            timer(start=start, end=end)

            pred = clf.predict(X_test)
            accuracy, f1, precision, recall = classification_metrics(
                y_true=y_test,
                y_pred=pred
            )
            trained_models[(f1, recall, precision, accuracy)] = clf
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
        accuracy, f1, precision, recall = classification_metrics(
            y_true=y,
            y_pred=pred
        )
        scores = {
            'Accuracy': accuracy,
            'F1 score': f1,
            'Precision': precision,
            'Recall': recall
        }
        return json.dumps(scores, indent=2, sort_keys=True)


class CutoRegressor:
    super(CutoRegressor, self).__init__()

    def __init__(self, test_size=0.2):
        self.models = regressors.models
        self.best_estimator = None
        self.test_size = test_size

    def fit(self, X, y):
        X_train, y_train, X_test, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=0
        )
        trained_models = dict()
        for model in self.models:
            clf = Pipeline([
                ('standard_scale', StandardScaler()),
                ('regression_model', model)
            ])
            start = time.time()
            clf.fit(X=X, y=y)
            end = time.time()
            timer(start=start, end=end)

            pred = clf.predict(X_test)
            r2, mape, mse, mae = regression_metrics(
                y_true=y_test,
                y_pred=pred
            )
            trained_models[(r2, mape, mse, mae)] = clf
        self.best_estimator = max(
            sorted(trained_models.items(), reverse=True))[1]

    def predict(self, X):
        prediction = self.best_estimator.predict(X)
        return prediction

    def score(self, X, y):
        assert self.best_estimator, "Models not fitted yet"
        pred = self.best_estimator.predict(X)
        r2, mape, mse, mae = regression_metrics(
            y_true=y_test,
            y_pred=pred
        )
        scores = {
            'R2 score': r2,
            'MAPE': mape,
            'MSE': mse,
            'MAE': mae
        }
        return json.dumps(scores, indent=2, sort_keys=True)
