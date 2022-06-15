from sklearn import datasets
from sklearn.model_selection import train_test_split
from cutoml.cutoml import CutoRegressor

if __name__ == "__main__":
    dataset = datasets.load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=0
    )

    ctr = CutoRegressor(k_folds=5, n_jobs=2, random_state=1)
    ctr.fit(X=X_train, y=y_train)
    print(f'R2 Score: {ctr.score(X=X_test, y=y_test)}')
    print(ctr.best_estimator.named_steps["regression_model"].best_estimator_)
