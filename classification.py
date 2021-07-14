from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from utils import *
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from best_hyperparamters import *


tests_num = 1


def check_knn(x_train, y_train, x_test, y_test, k):

    model = KNeighborsClassifier(n_neighbors=k)
    # Train the model using the training sets
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc_score = metrics.accuracy_score(y_test, y_pred)

    return acc_score


def check_decision_tree(x_train, y_train, x_test, y_test):

    # instantiate the DecisionTreeClassifier model with max_depth = 4
    clf = DecisionTreeClassifier(max_depth=4)

    # Train Decision Tree Classifer
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    return accuracy


def check_random_forest(x_train, y_train, x_test, y_test):

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(max_depth=4)

    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # prediction on test set
    y_pred = clf.predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)

    return accuracy_score


def check_naive_bayes(x_train, y_train, x_test, y_test):

    clf = MultinomialNB()
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)

    return accuracy_score


def run_ml_project():

    dh = DataHolder()

    for label in ["hypertension", "heart_disease", "stroke"]:
        best_performence_algo = {'knn': 0, 'naive_bayes': 0,
                                 'decision_tree': 0, 'random_forest': 0}
        features, labels = get_features_labels(dh, label)
        features = select_features(features, labels, chi2)

        print(f"\nall following results are for {label}")
        k = check_best_k_for_knn(features, labels)

        for _ in range(tests_num):
            x_train, x_test, y_train, y_test = split_data(features, labels)

            best_performence_algo['knn'] += check_knn(
                x_train, y_train, x_test, y_test, 13)
            best_performence_algo['naive_bayes'] += check_naive_bayes(
                x_train, y_train, x_test, y_test)
            best_performence_algo['decision_tree'] += check_decision_tree(
                x_train, y_train, x_test, y_test)
            best_performence_algo['random_forest'] += check_random_forest(
                x_train, y_train, x_test, y_test)

        max_algo = max(best_performence_algo,
                       key=best_performence_algo.get)
        max_perf = max(best_performence_algo.values())

        for key, val in best_performence_algo.items():

            print(key, "accuracy:", val / tests_num)

        print(
            f"\n{max_algo} has the best performances, with {round(100*(max_perf/tests_num), 2)}% accuracy!")


if __name__ == "__main__":
    run_ml_project()
