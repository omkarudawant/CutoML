from warnings import filterwarnings

filterwarnings("ignore")

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO

from core.pipeline import preprocess_pipeline
from core.trainer import Trainer
from core.util import timer
from sklearn.metrics import classification_report

import time
import uvicorn
import joblib
import pandas as pd
import numpy as np
import multiprocessing

app = FastAPI()


class ShortDescription(BaseModel):
    description: str


@app.post("/preprocess/")
def preprocess_data(features_file: UploadFile = File(...),
                    labels_file: UploadFile = File(...)):
    if features_file.filename.endswith((".csv", ".CSV")) and \
            labels_file.filename.endswith((".csv", ".CSV")):

        features_csv_file = features_file.file.read()
        labels_csv_file = labels_file.file.read()

        features_df = pd.read_csv(BytesIO(features_csv_file))
        labels_df = pd.read_csv(BytesIO(labels_csv_file))

        features_col_name = features_df.columns.tolist()[0]
        label_col_name = labels_df.columns.tolist()[0]

        df = pd.concat([features_df, labels_df], axis=1)
        output = preprocess_pipeline.transform(X=df)
        print(output.head(), features_col_name, label_col_name, output.shape)

        output[features_col_name].to_csv("data/preprocessed_features_data.csv",
                                         index=False)
        output[label_col_name].to_csv("data/preprocessed_labels_data.csv",
                                      index=False)
        return {
            "Message": "Preprocessed data saved to data/ directory"
        }
    else:
        return {
            "Message": "Please upload a csv features file and labels file"
        }


@app.post("/train/")
def train_model(features_file: UploadFile = File(...),
                labels_file: UploadFile = File(...)):
    if features_file.filename.endswith((".csv", ".CSV")) and \
            labels_file.filename.endswith((".csv", ".CSV")):

        features_csv_file = features_file.file.read()
        labels_csv_file = labels_file.file.read()

        features_df = pd.read_csv(BytesIO(features_csv_file))
        labels_df = pd.read_csv(BytesIO(labels_csv_file))

        features_col_name = features_df.columns.tolist()[0]
        label_col_name = labels_df.columns.tolist()[0]

        df = pd.concat([features_df, labels_df], axis=1)

        df = df.sample(frac=0.05)
        X = df[features_col_name].values.astype("U")
        y = df[label_col_name]

        model = Trainer(X=X, y=y)
        model.fit_models()
        predictions = model.best_estimator.predict(model.X_test)

        report = classification_report(
            y_true=model.y_test, y_pred=predictions,
            output_dict=True
        )

        pd.DataFrame(
            classification_report(
                y_true=model.y_test, y_pred=predictions,
                output_dict=True
            )
        ).T.to_csv("data/classification_report.csv")

        name_of_model = model.best_estimator.get_params()["classifier"]

        try:
            name_of_model = name_of_model.best_estimator_
            print(f"Best estimator: {name_of_model}")
        except Exception:
            print(f"Best estimator: {name_of_model}")
        accuracy, f1, precision, recall = model.score()
        return {
            "Model_name": name_of_model.__class__.__name__,
            "Accuracy": accuracy,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        }

    else:
        return {
            "Message": "Please upload a csv features file and labels file"
        }


@app.post("/predict_single/")
async def predict_asgn_group(short_desc: ShortDescription):
    short_desc_dict = short_desc.dict()
    model = None
    prediction_label = None

    if short_desc.description:
        start = time.time()
        try:
            model = joblib.load("models/train_pipeline.joblib")
            # TODO: Change filename to persistent storage path
            print(model)
            prediction_label = model.predict([short_desc.description])
            end = time.time()
            timer(start=start, end=end)
        except FileNotFoundError:
            print("Model not trained yet")

        try:
            prediction_probablity = model.predict_proba(
                [short_desc.description]
            )
            print(prediction_probablity)
            short_desc_dict.update({
                "Prediction confidence": str(round(np.max(
                    prediction_probablity), 4) * 100)
            })
        except AttributeError:
            print(model)

        print(prediction_label)

        prediction_enc = prediction_label[0]
        short_desc_dict.update({
            "Predicted assignment group": str(prediction_enc)
        })
    return short_desc_dict


@app.post("/predict_batch/")
async def batch_predictions(file: UploadFile = File(...)):
    try:
        model = None
        pred_asgn_grps = None
        if file.filename.endswith((".csv", ".CSV")):
            csv_file = await file.read()
            df = pd.read_csv(BytesIO(csv_file))
            df.dropna(inplace=True)

            X = df["short_descriptions"].values
            y = df["priority"].values

            start = time.time()
            try:
                model = joblib.load("models/train_pipeline.joblib")
                # TODO: Change filename to persistent storage path
                print(model)
                pred_asgn_grps = model.predict(X)
                end = time.time()
                timer(start=start, end=end)
            except FileNotFoundError:
                print("Model not trained yet")

            output = list()
            for short_desc, true_grp, pred_grp in zip(
                    X[:20], y[:20], pred_asgn_grps[:20]
            ):
                result = dict()
                result["short_descriptions"] = short_desc
                result["true_priority"] = true_grp
                result["pred_priority"] = pred_grp
                output.append(result)
            return output
        else:
            return {
                "Message": "Please upload a csv features_file"
            }

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, port=5000)
