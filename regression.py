from sklearn import ensemble
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from utils import *

from sklearn.metrics import mean_squared_error as MSE
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def linear_regression(dh, label):
    features, labels = get_features_labels(dh, label)
    features = select_features(features, labels)

    x_train, x_test, y_train, y_test = split_data(features, labels)

    LR = LinearRegression()
    LR.fit(x_train, y_train)
    y_pred = LR.predict(x_test)
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
    # r2 = LR.score(x_test, y_test)


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
    plot_output(reg, x_test, y_test, params)

    check_linear_regression(x_test, x_train, y_test, y_train)

    dt = DecisionTreeRegressor()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    mse_dt = MSE(y_test, y_pred)
    rmse_dt = mse_dt ** (1 / 2)
    print(
        f"accuracy for avg_glucose_level using DecisionTreeRegressor is:{mse_dt}")
    plot_output(dt, x_test, y_test)


    knnr = KNeighborsRegressor()
    knnr.fit(x_train, y_train)
    y_pred = knnr.predict(x_test)
    mse_knn = MSE(y_test, y_pred)
    rmse_dt = mse_knn**(1/2)
    print(
        f"accuracy for avg_glucose_level using KnnRegressor is:{mse_dt}")
    plot_output(mse_knn,  knnr, x_test, y_test)


def check_linear_regression(x_test, x_train, y_test, y_train):
    pcr = make_pipeline(StandardScaler(), PCA(
        n_components=1), LinearRegression())
    pcr.fit(x_train, y_train)
    pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline

    pls = PLSRegression(n_components=1)
    pls.fit(x_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].scatter(pca.transform(x_test), y_test,
                    alpha=.3, label='ground truth')
    axes[0].scatter(pca.transform(x_test), pcr.predict(x_test), alpha=.3,
                    label='predictions')
    axes[0].set(xlabel='Projected data onto first PCA component',
                ylabel='y', title='PCR / PCA')
    axes[0].legend()
    axes[1].scatter(pls.transform(x_test), y_test,
                    alpha=.3, label='ground truth')
    axes[1].scatter(pls.transform(x_test), pls.predict(x_test), alpha=.3,
                    label='predictions')
    axes[1].set(xlabel='Projected data onto first PLS component',
                ylabel='y', title='PLS')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    print(f"PCR r-squared {pcr.score(x_test, y_test):.3f}")
    print(f"PLS r-squared {pls.score(x_test, y_test):.3f}")
    pca_2 = make_pipeline(PCA(n_components=2), LinearRegression())
    pca_2.fit(x_train, y_train)
    print(f"PCR r-squared with 2 components {pca_2.score(x_test, y_test):.3f}")



def plot_output(y_pred, y_test, name):
    plt.figure(figsize=(4, 3))
    plt.scatter(y_test, y_pred)
    plt.plot([ 0, 50 ], [ 0, 50 ], '--k')
    plt.title(type(reg))
    plt.axis('tight')
    plt.xlabel('True avg_glucose_level ')
    plt.ylabel('Predicted avg_glucose_level ')
    plt.tight_layout()
    plt.show()


def get_tests_scores(params, reg, x_test, y_test):
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(x_test)):
        test_score[i] = reg.loss_(y_test, y_pred)
    return y_pred



def vote_results(dh):
    features, labels = get_features_labels(dh,  "avg_glucose_level")
    X, x_test, y, y_test = split_data(features, labels, 0.01)

    # Train classifiers
    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = RandomForestRegressor(random_state=1)
    reg3 = LinearRegression()

    dt = DecisionTreeRegressor(max_depth = 2)
    dt.fit(X, y)
    pred_dt = dt.predict(x_test)
    print_score(pred_dt, y_test, "Decision tree max_depth =2", x_test)

    reg1.fit(X, y)
    reg2.fit(X, y)
    reg3.fit(X, y)

    ereg = VotingRegressor([ ('gb', reg1), ('rf', reg2), ('lr', reg3), ('dt', dt) ])
    ereg.fit(X, y)

    pred1 = reg1.predict(x_test)
    print_score(pred1, y_test, "gradient boosting", x_test)
    pred2 = reg2.predict(x_test)
    print_score(pred2, y_test, "random forest", x_test)

    pred3 = reg3.predict(x_test)
    print_score(pred3, y_test, "linear regression", x_test)

    pred4 = ereg.predict(x_test)
    print_score(pred1, y_test, "voting", x_test)


    plt.figure()
    plt.plot(pred1, 'gd', label='GradientBoostingRegressor')
    plt.plot(pred2, 'b^', label='RandomForestRegressor')
    plt.plot(pred3, 'ys', label='LinearRegression')
    plt.plot(pred4, 'r*', ms=10, label='VotingRegressor')
    plt.plot(y_test.to_numpy(), '.', label= "real value" )

    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.ylabel('predicted')
    plt.xlabel('training samples')
    plt.legend(loc="best")
    plt.title('Regressor predictions and their average')

    plt.show()


def print_score(y_pred, y_test, name, x_test):
    print(f"{name} explained variannce score: {explained_variance_score(y_test, y_pred)}")
    print(f"max error {name}:{max_error(y_test, y_pred)} ")
    print(f"mean abs error {name}:{mean_absolute_error(y_test, y_pred)} ")
    print(f"mean sqrd error {name}:{mean_squared_error(y_test, y_pred)} ")
    print(f"mean_squared_log_error {name}: {mean_squared_log_error(y_test, y_pred)}")
    print(f"mean_absolute_percentage_error {name}: {mean_absolute_percentage_error(y_test, y_pred)}")
    print(f"median_absolute_error {name}: {median_absolute_error(y_test, y_pred)}")
    print(f"r2 score {name}: {r2_score(y_test, y_pred)}")
    plot_output(y_pred, y_test, name)
    print()






if __name__ == "__main__":

    dh = DataHolder()
    # vote_results(dh)
    # check_dt_reg(dh, "avg_glucose_level")
    linear_regression(dh, "avg_glucose_level")
