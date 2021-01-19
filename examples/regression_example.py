from warnings import filterwarnings

filterwarnings('ignore')

from sklearn import datasets
from sklearn.model_selection import train_test_split
from cutoml.cutoml import CutoRegressor

dataset = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    test_size=0.2)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

ctr = CutoRegressor()
ctr.fit(X=X_train, y=y_train)
print(ctr.score(X=X_test, y=y_test))
