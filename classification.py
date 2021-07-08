from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn import metrics
from utils import DataHolder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def split_data(data, labels):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data and it's labels columns
    :return: train set, test set , labels of training and test sets data frames
    """
    return train_test_split(data, labels, test_size=0.5)


def knn(x_train, y_train, x_test, n_neighbors=71):

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Train the model using the training sets
    model.fit(x_train, y_train)

    return model.predict(x_test)


def check_knn(dh, label):
    avgs_dict = {}
    features, labels = get_features_labels(dh, label)
    splits_num = 100
    for _ in range(splits_num):
        x_train, x_test, y_train, y_test = split_data(features, labels)
        for i in range(1, 73):
            y_pred = knn(x_train, y_train, x_test, i)
            acc_score = metrics.accuracy_score(y_test, y_pred)
            avgs_dict[i] = avgs_dict.get(i, 0) + acc_score
    max_key = max(avgs_dict, key=avgs_dict.get)
    max_val = max(avgs_dict.values())
    print(
        f"best accuracy for {label} is for {max_key} neighbours: {max_val / splits_num}")


def check_decision_tree(dh, label):

    features, labels = get_features_labels(dh, label)
    splits_num = 100
    avg_gini = 0
    avg_en = 0
    for _ in range(splits_num):
        x_train, x_test, y_train, y_test = split_data(features, labels)

        # instantiate the DecisionTreeClassifier model with criterion gini index
        clf_gini = DecisionTreeClassifier(criterion='gini')

        # instantiate the DecisionTreeClassifier model with criterion gini index
        clf_en = DecisionTreeClassifier(criterion='entropy')
        # Train Decision Tree Classifer
        clf_gini.fit(x_train, y_train)
        clf_en.fit(x_train, y_train)

        # Predict the response for test dataset
        y_pred_gini = clf_gini.predict(x_test)
        y_pred_en = clf_en.predict(x_test)

        avg_gini += metrics.accuracy_score(y_test, y_pred_gini)
        avg_en += metrics.accuracy_score(y_test, y_pred_en)
    max_criterion = "gini" if (avg_gini > avg_en) else "entropy"

    print(
        f"decision tree: accuracy for {label} with gini criterion: {avg_gini/splits_num}, with entropy criterion: {avg_en/splits_num}, best perfomance: {max_criterion}   ")


def check_naive_bayes(dh, label):
    totals = {"gaussian": 0, "categorial": 0, "multinomial": 0}
    features, labels = get_features_labels(dh, label)
    splits_num = 100
    for _ in range(splits_num):
        x_train, x_test, y_train, y_test = split_data(features, labels)
        gnb = GaussianNB()
        y_pred = gnb.fit(x_train, y_train).predict(x_test)
        acc_score = metrics.accuracy_score(y_test, y_pred)
        totals["gaussian"] += acc_score
        clf = MultinomialNB()
        y_pred = clf.fit(x_train, y_train).predict(x_test)
        acc_score = metrics.accuracy_score(y_test, y_pred)
        totals["multinomial"] += acc_score
    max_key = max(totals, key=totals.get)
    max_val = max(totals.values())
    print(
        f"avg naive bayes accuracy for {label} is {max_key} : {max_val / splits_num}")


def get_features_labels(dh, label):
    features = dh.get_features(label)
    labels = dh.get_labels(label)
    return features, labels


# def check_accuracy():
if __name__ == "__main__":

    dh = DataHolder()
    for label in ["hypertension", "heart_disease", "stroke"]:
        check_knn(dh, label)
        check_naive_bayes(dh, label)
        check_decision_tree(dh, label)


# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# 5110 samples (in knn: k = 71)
