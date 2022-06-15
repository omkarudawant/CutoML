<img src="https://user-images.githubusercontent.com/51676294/105076115-ac83e100-5ab0-11eb-8743-9a6dcc21cf45.png" width="90%"></img>


[![PyPI version](https://badge.fury.io/py/cutoml.svg)](https://pypi.org/project/cutoml/)
[![Downloads](https://pepy.tech/badge/cutoml)](https://pepy.tech/project/cutoml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/omkarudawant/CutoML/blob/main/LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/omkarudawant/CutoML/pulls)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/omkarudawant/CutoML)

CutoML is a lightweight automl library, highly optimized to give you the best possible model depending on your datasets very quickly.


Installation
------------

    pip install -U cutoml

Usage Example
-------------


For classification,

```python

from cutoml.cutoml import CutoClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

dataset = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.25)

ctc = CutoClassifier(k_folds=5, n_jobs=2)
ctc.fit(X=X_train, y=y_train)
```

For regression,

```python

from cutoml.cutoml import CutoRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets

dataset = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.25)

ctr = CutoRegressor(k_folds=5, n_jobs=2)
ctr.fit(X=X_train, y=y_train)
```
