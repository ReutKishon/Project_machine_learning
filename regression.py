import csv
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from utils import *
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_percentage_error, median_absolute_error, \
    mean_squared_log_error, mean_absolute_error
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import pandas as pd

"""
        plot columns graph which compare between actual values and predicted values of test set.
       :param y_test: actual values
       :param y_pred: predicted values

"""


def plot_columns_graph_output(y_test, y_pred):
    y_test = np.array(list(y_test))
    y_pred = np.array(y_pred)
    df = pd.DataFrame({'Actual': y_test.flatten(),
                      'Predicted': y_pred.flatten()})
    print(df)
    df1 = df.head(25)
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()


def check_best_k_for_knn_regressor(x_train, x_test, y_train, y_test):

    rmse_val = []  # to store rmse values for different k
    min_k = 0
    min_rmse = float("inf")

    for K in range(1, 73, 2):

        model = KNeighborsRegressor(n_neighbors=K)

        model.fit(x_train, y_train)  # fit the model
        pred = model.predict(x_test)  # make prediction on test set
        error = sqrt(mean_squared_error(y_test, pred))  # calculate rmse
        rmse_val.append(error)  # store rmse values
        if (error < min_rmse):
            min_rmse = error
            min_k = K

    # plotting the rmse values against k values
    curve = pd.DataFrame(rmse_val, columns=["Validation error"])  # elbow curve
    curve.plot()

    plt.xlabel('K-value')
    plt.ylabel('RMSE')
    plt.show()

    return min_k


def knn_regressor(x_train, x_test, y_train, K):

    # Create a KNeighborsRegressor Classifier
    model = KNeighborsRegressor(n_neighbors=K)
    # Train the model using the training sets
    model.fit(x_train, y_train)  # fit the model
    y_pred = model.predict(x_test)  # make prediction on test set
    return y_pred


def linear_regression(x_train, x_test, y_train):

    LR = LinearRegression()
    LR.fit(x_train, y_train)  # fit the model
    y_pred = LR.predict(x_test)  # make prediction on test set
    return y_pred


def decision_tree_regressor(x_train, x_test, y_train):
    # instantiate the DecisionTreeRegressor model with max_depth = 4
    dt = DecisionTreeRegressor(max_depth=4)
    dt.fit(x_train, y_train)  # fit the model
    y_pred = dt.predict(x_test)  # make prediction on test set
    return y_pred


def random_forest_regressor(x_train, x_test, y_train):
    rf = RandomForestRegressor(random_state=1, max_depth=4)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return y_pred


def plot_output(y_pred, y_test, name):
    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, y_pred)
    plt.plot([0, 50], [0, 50], '--k')
    plt.title(name)
    plt.axis('tight')
    plt.xlabel('True avg_glucose_level ')
    plt.ylabel('Predicted avg_glucose_level ')
    plt.tight_layout()
    plt.show()


def create_csv_file():
    with open('results.csv', 'w',  newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["algorithm", "explained_variance_score", "max_error", "mean_absolute_error", "mean_squared_error", "mean_squared_log_error", "mean_absolute_percentage_error",
                             "median_absolute_error", "r2"])

# insert performence results of one algorithm to excel file


def insert_row_into_excel(name, y_test, y_pred,):
    scores = [name, explained_variance_score(y_test, y_pred), max_error(y_test, y_pred), mean_absolute_error(y_test, y_pred),
              mean_squared_error(y_test, y_pred), mean_squared_log_error(
                  y_test, y_pred),
              mean_absolute_percentage_error(
                  y_test, y_pred), median_absolute_error(y_test, y_pred),
              r2_score(y_test, y_pred)]

    with open('results.csv', 'a',  newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(scores)


def results_in_excel_file(x_train, x_test, y_train, y_test):
    k = check_best_k_for_knn_regressor(x_train, x_test, y_train, y_test)
    knn_pred = knn_regressor(x_train, x_test, y_train,  k)
    lr_pred = linear_regression(x_train, x_test, y_train)
    dt_pred = decision_tree_regressor(x_train, x_test, y_train)
    rf_pred = random_forest_regressor(x_train, x_test, y_train)
    insert_row_in_excel("Knn", y_test, knn_pred)
    insert_row_in_excel("linear regression", y_test, lr_pred)
    insert_row_in_excel("decision tree", y_test, dt_pred)
    insert_row_in_excel("random forest", y_test, rf_pred)


# def print_score(y_pred, y_test, name, x_test):

#     print(
#         f"explained variannce score {name}: {explained_variance_score(y_test, y_pred)}")
#     print(f"max error {name}:{max_error(y_test, y_pred)} ")
#     print(f"mean abs error {name}:{mean_absolute_error(y_test, y_pred)} ")
#     print(f"mean sqrd error {name}:{mean_squared_error(y_test, y_pred)} ")
#     print(
#         f"mean_squared_log_error {name}: {mean_squared_log_error(y_test, y_pred)}")
#     print(
#         f"mean_absolute_percentage_error {name}: {mean_absolute_percentage_error(y_test, y_pred)}")
#     print(
#         f"median_absolute_error {name}: {median_absolute_error(y_test, y_pred)}")
#     print(f"r2 score {name}: {r2_score(y_test, y_pred)}")


if __name__ == "__main__":

    create_csv_file()
    dh = DataHolder()
    features, labels = get_features_labels(dh, "avg_glucose_level")
    x_train, x_test, y_train, y_test = split_data(features, labels)

    result_in_excel_file(x_train, x_test, y_train, y_test)
