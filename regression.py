from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import DataHolder
from utils import split_data
from utils import get_features_labels
from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot as plt
import numpy as np


def check_dt_reg(dh, label):
    features, labels = get_features_labels(dh, label)
    x_train, x_test, y_train, y_test = split_data(features, labels)

    params = {'n_estimators': 500,
              'max_depth': 4,
              'min_samples_split': 5,
              'learning_rate': 0.01,
              'loss': 'ls'}
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(x_train, y_train)
    mse = MSE(y_test, reg.predict(x_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(reg.staged_predict(x_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 50], [0, 50], '--k')
    plt.axis('tight')
    plt.xlabel('True avg_glucose_level ')
    plt.ylabel('Predicted avg_glucose_level ')
    plt.tight_layout()
    plt.show()

    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)

    y_pred = dt.predict(x_test)

    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 50], [0, 50], '--k')
    plt.axis('tight')
    plt.xlabel('True avg_glucose_level ')
    plt.ylabel('Predicted avg_glucose_level ')
    plt.tight_layout()
    plt.show()

    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt**(1/2)
    print(
        f"accuracy for avg_glucose_level using DecisionTreeRegressor is:{mse_dt}")

    knnr = KNeighborsRegressor()
    knnr.fit(x_train, y_train)
    y_pred = knnr.predict(x_test)

    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 50], [0, 50], '--k')
    plt.axis('tight')
    plt.xlabel('True avg_glucose_level ')
    plt.ylabel('Predicted avg_glucose_level ')
    plt.tight_layout()
    plt.show()

    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt**(1/2)
    print(
        f"accuracy for avg_glucose_level using KnnRegressor is:{mse_dt}")


if __name__ == "__main__":

    dh = DataHolder()
    check_dt_reg(dh, "avg_glucose_level")
