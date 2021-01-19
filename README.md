<img src="https://user-images.githubusercontent.com/51676294/105076115-ac83e100-5ab0-11eb-8743-9a6dcc21cf45.png" width="90%"></img> 

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) 
[![PyPI version](https://badge.fury.io/py/cutoml.svg)](https://badge.fury.io/py/cutoml) 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) 
[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/omkarudawant/CutoML/pulls)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/omkarudawant/StrapDown.js/graphs/commit-activity)


A lightweight automl framework for classification/regression tasks.


Installation
------------

    pip install cutoml

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
                                                    test_size=0.2)

ctc = CutoClassifier()
ctc.fit(X=X_train, y=y_train)
```

For classification,
  
```python
from cutoml.cutoml import CutoRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets

dataset = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.2)

ctr = CutoRegressor()
ctr.fit(X=X_train, y=y_train)
```

