import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from cutoml.cutoml import CutoClassifier

if __name__ == "__main__":
    dataset = datasets.load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.3
    )

    ctc = CutoClassifier(k_folds=3, n_jobs=4, verbose=0)
    ctc.fit(X=X_train, y=y_train)
    print(ctc.score(X=X_test, y=y_test))
    print(ctc.best_estimator.named_steps["classification_model"].best_estimator_)
