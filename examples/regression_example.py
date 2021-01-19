import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from sklearn import datasets
from sklearn.model_selection import train_test_split
from cutoml.cutoml import CutoRegressor

dataset = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.2)

ctr = CutoRegressor(k_folds=3, n_jobs=2, verbose=1)
ctr.fit(X=X_train, y=y_train)
print(ctr.score(X=X_test, y=y_test))
print(ctr.best_estimator.get_params()['regression_model'].best_estimator_)
