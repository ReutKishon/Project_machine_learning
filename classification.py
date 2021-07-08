from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn import metrics
from utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# searching for the best k for knn algo


def check_best_k(features, labels):
    options_for_k = {k: 0 for k in range(1, 73)}

    for i in range(100):
        x_train, x_test, y_train, y_test = split_data(
            features, labels)

        for k in range(1, 73):
            acc_score = check_knn(x_train, y_train, x_test, y_test, k)
            options_for_k[k] += acc_score
    max_key = max(options_for_k, key=options_for_k.get)
    return max_key


def check_knn(x_train, y_train, x_test, y_test, k):

    model = KNeighborsClassifier(n_neighbors=k)
    # Train the model using the training sets
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    acc_score = metrics.accuracy_score(y_test, y_pred)
    return acc_score


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


# def check_accuracy():
if __name__ == "__main__":

    dh = DataHolder()
    for label in ["hypertension", "heart_disease", "stroke"]:
        features, labels = get_features_labels(dh, label)
        print(check_best_k(features, labels))

        # for i in range(100):
        #     x_train, x_test, y_train, y_test = split_data(features, labels)

        #     check_knn(x_train, y_train, x_test, y_test)
        #     check_naive_bayes(x_train, y_train, x_test, y_test)
        #     check_decision_tree(x_train, y_train, x_test, y_test)
        #     check_random_forest(x_train, y_train, x_test, y_test)


# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# Knn : the classification result depends on choosing k and that's the difficult task in knn.
# 5110 samples (in knn: k = 71)
