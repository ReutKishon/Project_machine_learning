from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn import metrics
from utils import DataHolder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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


def check_knn(x_train, y_train, x_test, y_test):
    avgs_dict = {}
    splits_num = 100
    for i in range(1, 73):
        y_pred = knn(x_train, y_train, x_test, i)
        acc_score = metrics.accuracy_score(y_test, y_pred)
        avgs_dict[i] =  acc_score
    max_key = max(avgs_dict, key=avgs_dict.get)
    max_val = max(avgs_dict.values())
    print(
        f"best accuracy for {label} is for {max_key} neighbours: {max_val}")


def check_decision_tree(x_train, y_train, x_test, y_test):

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

    accuracy_gini = metrics.accuracy_score(y_test, y_pred_gini)
    accuracy_en = metrics.accuracy_score(y_test, y_pred_en)
    max_criterion = "gini" if (accuracy_gini > accuracy_en) else "entropy"

    print(
        f"decision tree: accuracy for {label} with gini criterion: {accuracy_gini}, with entropy criterion: {accuracy_en}, best perfomance: {max_criterion}   ")


def check_random_forest(x_train, y_train, x_test, y_test):

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(x_train, y_train)

    # prediction on test set
    y_pred = clf.predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)

    print(
        f"Random forest accuracy for {label} is {accuracy_score}")


def check_naive_bayes(x_train, y_train, x_test, y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    gaussian_score = metrics.accuracy_score(y_test, y_pred)
    clf = MultinomialNB()
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    multinomial_score = metrics.accuracy_score(y_test, y_pred)
    max_key = "gaussian" if (
        gaussian_score > multinomial_score) else "multinomial"
    max_val = max(gaussian_score, multinomial_score)
    print(
        f"avg naive bayes accuracy for {label} is {max_key} : {max_val}")


def get_features_labels(dh, label):
    features = dh.get_features(label)
    labels = dh.get_labels(label)
    return features, labels


# def check_accuracy():
if __name__ == "__main__":

    dh = DataHolder()

    for label in ["hypertension", "heart_disease", "stroke"]:
        features, labels = get_features_labels(dh, label)
        x_train, x_test, y_train, y_test = split_data(features, labels)

        check_knn(x_train, y_train, x_test, y_test)
        check_naive_bayes(x_train, y_train, x_test, y_test)
        check_decision_tree(x_train, y_train, x_test, y_test)
        check_random_forest(x_train, y_train, x_test, y_test)


# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# Knn : the classification result depends on choosing k and that's the difficult task in knn.
# 5110 samples (in knn: k = 71)
