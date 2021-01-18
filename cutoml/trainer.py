from cutoml.model_list import models
from cutoml.util import calculate_metrics
from cutoml.util import timer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

import pandas as pd
import time
import multiprocessing
import joblib


class Trainer:
    def __init__(self, X, y, test_size=0.2):
        super(Trainer, self).__init__()
        self.X = X
        self.y = y
        self.test_size = test_size
        self.encoder = LabelEncoder()
        self.y_enc = self.encoder.fit_transform(self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, self.y_enc, test_size=self.test_size, random_state=0
        )

        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
            self.X_test, self.y_test, test_size=0.5, random_state=0
        )
        print('Dataset splits:')
        print(
            f'Train: {self.X_train.shape}',
            f'Test: {self.X_test.shape}',
            f'Validation: {self.X_val.shape}'
        )
        print('Saving validation data...')
        pd.DataFrame(
            {
                "short_descriptions": self.X_val,
                "priority": self.y_val
            }
        ).to_csv(
            "data/validation_set.csv",
            index=False
        )
        print('Done !')
        self.trained_models = dict()
        self.models = models
        self.best_estimator = None

    def fit_models(self):
        print('-' * 50)
        for model in self.models:
            print(f'Model: {model}')
            classifier = Pipeline([
                ('hash_vectorize', HashingVectorizer(
                    n_features=2 ** 15,
                    alternate_sign=False)
                 ),
                ('over_sample_with_smote', SMOTE(
                    n_jobs=multiprocessing.cpu_count() // 2,
                    random_state=0)
                 ),
                ('standard_scaling', StandardScaler(with_mean=False)),
                ('classifier', model)
            ])
            start_time = time.time()
            classifier.fit(self.X_train, self.y_train)
            end_time = time.time()
            timer(start=start_time, end=end_time)

            predictions = classifier.predict(self.X_test)

            acc, f1, precision, recall = calculate_metrics(
                y_true=self.y_test,
                y_pred=predictions
            )
            self.trained_models[f1] = classifier
            print('-' * 100)
        self.best_estimator = max(
            sorted(self.trained_models.items(), reverse=True))[1]
        joblib.dump(
            value=self.best_estimator, filename="models/train_pipeline.joblib",
            compress=5
        )
        joblib.dump(
            value=self.encoder, filename="models/encoder.joblib",
            compress=5
        )  # TODO: Change filename to persistent storage path

    def score(self):
        assert self.best_estimator, "Models not fit yet"
        predictions = self.best_estimator.predict(self.X_test)
        accuracy, f1, precision, recall = calculate_metrics(
            y_true=self.y_test,
            y_pred=predictions
        )
        return accuracy, f1, precision, recall
