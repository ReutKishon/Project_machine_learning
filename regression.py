from sklearn.tree import DecisionTreeRegressor
from utils import DataHolder
from utils import split_data
from utils import get_features_labels
from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot as plt


def check_dt_reg(dh, label):
    features, labels = get_features_labels(dh, label)
    x_train, x_test, y_train, y_test = split_data(features, labels)
    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    for i in y_pred[:12]:
        print(i)

    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 50], [0, 50], '--k')
    plt.axis('tight')
    plt.xlabel('True avg_glucose_level ')
    plt.ylabel('Predicted avg_glucose_level ')
    plt.tight_layout()
    plt.show()

# mse_dt = MSE(y_test, y_pred)
# rmse_dt = mse_dt**(1/2)
# print(
#     f"accuracy for avg_glucose_level using DecisionTreeRegressor is:{rmse_dt}")


if __name__ == "__main__":

    dh = DataHolder()
    check_dt_reg(dh, "avg_glucose_level")
