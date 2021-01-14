from warnings import filterwarnings

filterwarnings("ignore")

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO

from ticket_prioritization.pipeline import preprocess_pipeline
from ticket_prioritization.processors import display_metrics
from imblearn.pipeline import Pipeline as Imblearn_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import time
import uvicorn
import joblib
import pandas as pd
import numpy as np
import multiprocessing


def train_model(file):
    if file.filename.endswith((".csv", ".CSV")):
        
        df = pd.read_csv(file)

        X = df["short_descriptions"].values.astype("U")
        y = df["priority"]

        # enc = LabelEncoder()
        # y_enc = enc.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=0
        )

        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.5, random_state=0
        )

        print(
            X_train.shape,
            y_train.shape,
            X_test.shape,
            y_test.shape,
            X_val.shape,
            y_val.shape,
        )

        val_df = pd.DataFrame(
            {
                "short_descriptions": X_val,
                "priority": y_val
            }
        ).to_csv(
            "data/validation_set.csv",
            index=False
        )
        cross_val = 3
        models = [
            MultinomialNB(),
            # GridSearchCV(
            #     LogisticRegression(
            #         n_jobs=multiprocessing.cpu_count() // 2,
            #         random_state=0),
            #     param_grid={
            #         'penalty': ['l1', 'l2'],
            #         'C': [0.1, 1, 10, 100, 1000],
            #         'max_iter': [100, 300, 500, 700]
            #     },
            #     cv=cross_val
            # ),
            GridSearchCV(
                LinearSVC(random_state=0),
                param_grid={'C': [0.1, 1]}
            ),
            # GridSearchCV(
            #     RandomForestClassifier(
            #         n_jobs=multiprocessing.cpu_count() // 2,
            #         random_state=0),
            #     param_grid={
            #         'n_estimators': np.arange(start=100, stop=1000,
            #         step=200),
            #         'max_features': ['auto', 'sqrt', 'log2']},
            #     cv=cross_val
            # ),
            # GridSearchCV(
            #     XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2,
            #                   random_state=0),
            #     param_grid={
            #         'n_estimators': np.arange(start=100, stop=1000,
            #         step=200),
            #         'max_depth': np.arange(start=2, stop=10, step=2)},
            #     cv=cross_val
            # ),
            GridSearchCV(
                DecisionTreeClassifier(random_state=0),
                param_grid={'criterion': ['gini', 'entropy'],
                            'max_depth': np.arange(start=2, stop=7,
                                                   step=2)},
                cv=cross_val
            )
        ]

        trained_models = dict()
        print("-" * 50)
        for model in models:
            print(f"Training -> {model}")
            predictor = Imblearn_pipeline(
                [
                    (
                        "hash_vectorizing", HashingVectorizer(
                            n_features=2 ** 15,
                            alternate_sign=False
                        )
                    ),
                    (
                        "smote_over_sampling", SMOTE(random_state=0,
                                                     n_jobs=-1)
                    ),
                    (
                        "classification_model_fit", model
                    ),
                ]
            )

            s = time.time()
            predictor.fit(X_train, y_train)
            e = time.time()
            print(f"Training time: {round(e - s)} seconds")

            preds = predictor.predict(X_test)
            acc, f1, precision, recall = display_metrics(true=y_test,
                                                         pred=preds)

            trained_models[f1] = predictor
            print("-" * 50)

        final_model = max(sorted(trained_models.items(), reverse=True))[1]
        try:
            if final_model.get_params()[
                "classification_model_fit"].best_estimator_.__class__ \
                    .__name__ == 'LinearSVC':
                print('SVM it is !')

                final_model = CalibratedClassifierCV(
                    final_model.get_params()["classification_model_fit"])
                final_model.fit(X_train, y_train)

        except Exception:
            pass
        print("Best model: ", final_model)

        y_hat = final_model.predict(X_test)

        acc, f1, precision, recall = display_metrics(true=y_test, pred=y_hat)

        joblib.dump(
            value=final_model, filename="models/train_pipeline.joblib",
            compress=4
        )  # TODO: Change filename to persistent storage path

        report = classification_report(
            y_true=y_test, y_pred=final_model.predict(X_test), output_dict=True
        )
        report_df = pd.DataFrame(report)

        report_df.T.to_csv("data/classification_report.csv")

        name_of_model = final_model.get_params()[
            "classification_model_fit"]
        try:
            name_of_model = name_of_model.best_estimator_
            print(name_of_model)
            print(f"Pipeline params: {name_of_model}")
        except Exception:
            print('Problem !!!!!!!!!!')
            print(f"Pipeline params: {name_of_model}")

        return {
            "Model_name": name_of_model.__class__.__name__,
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        }

    else:
        return {
            "Message": "Please upload a csv file"
        }

if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser(
        'Script for facial recognition threshold testing\n')
    arg_parser.add_argument('--file_path',
                            help='Str, File Path to the Model training',
                            required=True)

    args = arg_parser.parse_args()
    result = train_model(file=args.file_path)
    print(result)
