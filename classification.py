from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn import metrics
from utils import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# searching for the best k for knn algo
tests_num = 100

def check_best_k_for_knn(features, labels):
    options_for_k = {k: 0 for k in range(1, 73)}

    for _ in range(tests_num):
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


def check_best_criterion_for_decision_tree(features, labels):

    options_for_criterion = {'gini': 0, 'entropy': 0}

    for _ in range(tests_num):
        x_train, x_test, y_train, y_test = split_data(
            features, labels)

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
        options_for_criterion['gini'] += accuracy_gini
        options_for_criterion['entropy'] += accuracy_en
    max_criterion = max(options_for_criterion, key=options_for_criterion.get)

    return max_criterion


def check_decision_tree(x_train, y_train, x_test, y_test, criterion):

    # instantiate the DecisionTreeClassifier model with criterion
    clf = DecisionTreeClassifier(criterion=criterion)

    # Train Decision Tree Classifer
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def check_random_forest(x_train, y_train, x_test, y_test):

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(x_train, y_train)

    # prediction on test set
    y_pred = clf.predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    return accuracy_score


def best_option_for_naive_bayes(features, labels):

    dic = {'GaussianNB': 0, 'multinomialNB': 0}

    for _ in range(tests_num):
        x_train, x_test, y_train, y_test = split_data(
            features, labels)

        gnb = GaussianNB()
        y_pred = gnb.fit(x_train, y_train).predict(x_test)
        dic['GaussianNB'] += metrics.accuracy_score(y_test, y_pred)
        clf = MultinomialNB()
        y_pred = clf.fit(x_train, y_train).predict(x_test)
        dic['multinomialNB'] += metrics.accuracy_score(y_test, y_pred)

    max_key = max(dic,
                  key=dic.get)

    return max_key


def check_naive_bayes(x_train, y_train, x_test, y_test, option):

    clf = GaussianNB() if (option == 'GaussianNB') else MultinomialNB()
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    return accuracy_score


def run_ml_project():
    dh = DataHolder()
    for label in [ "hypertension", "heart_disease", "stroke" ]:
        best_performence_algo = {'knn': 0, 'naive_bayes': 0,
                                 'decision_tree': 0, 'random_forest': 0}
        features, labels = get_features_labels(dh, label)
        k = check_best_k_for_knn(features, labels)
        print(f"all follwing results are for {label}")
        print(f"best k for knn algo is: {k}")
        criterion = check_best_criterion_for_decision_tree(features, labels)
        print(f"best criterion for decision_tree algo is: {criterion}")
        option_nb = best_option_for_naive_bayes(features, labels)
        print(f"best option for naive_bayes algo is: {option_nb}")

        for _ in range(tests_num):
            x_train, x_test, y_train, y_test = split_data(features, labels)

            best_performence_algo [ 'knn' ] += check_knn(
                x_train, y_train, x_test, y_test, k)
            best_performence_algo [ 'naive_bayes' ] += check_naive_bayes(
                x_train, y_train, x_test, y_test, option_nb)
            best_performence_algo [ 'decision_tree' ] += check_decision_tree(
                x_train, y_train, x_test, y_test, criterion)
            best_performence_algo [ 'random_forest' ] += check_random_forest(
                x_train, y_train, x_test, y_test)

        max_algo = max(best_performence_algo,
                       key=best_performence_algo.get)
        max_perf = max(best_performence_algo.values())
        # Iterating over values
        for key, val in best_performence_algo.items():
            print(key, "accuracy:", val / tests_num)
        print(f"{max_algo} has the best performances, with {max_perf} accuracy!")


if __name__ == "__main__":
    run_ml_project()

# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# Knn : the classification result depends on choosing k and that's the difficult task in knn.
# 5110 samples (in knn: k = 71)
