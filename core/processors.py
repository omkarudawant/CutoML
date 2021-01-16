from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import re
import string
import spacy

nlp = spacy.load("en_core_web_sm")


def display_metrics(true, pred):
    f1 = round(f1_score(y_true=true, y_pred=pred, average="weighted") * 100, 2)

    precision = round(
        precision_score(y_true=true, y_pred=pred, average="weighted") * 100, 2)

    recall = round(
        recall_score(y_true=true, y_pred=pred, average="weighted") * 100, 2)

    acc = round(accuracy_score(y_true=true, y_pred=pred) * 100, 2)

    print(f"Accuracy: {acc} | "
          f"F1: {f1} | "
          f"Precision: {precision} | "
          f"Recall: {recall}")

    return acc, f1, precision, recall


class CleanText(BaseEstimator, TransformerMixin):
    def __init__(self, feature_column_name, label_column_name):
        super(CleanText, self).__init__()
        self.label_column_name = label_column_name
        self.feature_column_name = feature_column_name
        self.replacements = {
            "4 - Low": "LOW",
            "5 - Minor": "LOW",
            "3 - Moderate": "MEDIUM",
            "2 - Normal": "MEDIUM",
            "Priority 4": "MEDIUM",
            "Priority 3": "HIGH",
            "2 - High": "HIGH",
            "Priority 2": "HIGH",
            "Priority 1": "CRITICAL",
            "1 - Critical": "CRITICAL",
            "1 - Urgent": "CRITICAL",
            "PROJECT": np.nan,
            "Service Request": np.nan,
        }

    # noinspection PyMethodMayBeStatic
    def _clean_text(self, text: str):
        text = text.lower()
        text = " ".join(text.split())  # remove \n \t
        text = re.sub("\s+", " ", text)
        text = re.sub(r"\w*\d\w*", "", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"  +", " ", text)
        return text

    # noinspection PyMethodMayBeStatic
    def preprocess_text(self, text: str):
        sentence = list()
        doc = nlp(text)
        for word in doc:
            sentence.append(word.lemma_)
        return " ".join(sentence)

    def clean_text(self, df: pd.DataFrame):
        dataframe = df.copy()

        tqdm.pandas()

        print("Before: ", dataframe[self.label_column_name].unique())

        dataframe = dataframe.replace(
            {self.label_column_name: self.replacements})
        dataframe = dataframe[~dataframe.index.duplicated()]
        dataframe.dropna(inplace=True)

        print("After: ", dataframe[self.label_column_name].unique())

        dataframe[self.label_column_name] = dataframe[
            self.label_column_name].progress_apply(lambda x: str(x))

        dataframe[self.feature_column_name] = dataframe[
            self.feature_column_name].progress_apply(
            lambda x: self._clean_text(str(x)))

        dataframe[self.feature_column_name] = dataframe[
            self.feature_column_name].progress_apply(lambda x:
                                                     self.preprocess_text(x))

        dataframe[self.feature_column_name] = dataframe[
            self.feature_column_name].progress_apply(lambda x: str(x))

        dataframe[self.feature_column_name] = dataframe[
            self.feature_column_name][dataframe[
            self.feature_column_name].progress_apply(
            lambda x: len(x.split()) > 2)]

        dataframe.dropna(inplace=True)

        dataframe = dataframe.sample(frac=1)
        return dataframe

    def fit(self, X, y) -> "CleanText":
        return self

    def transform(self, X):
        X = X.copy()
        print(f"Entered CleanText: {X.shape}")
        cleaned_dataframe = self.clean_text(df=X)
        X = cleaned_dataframe.copy()
        print(f"Done CleanText: {X.shape}")
        return X
