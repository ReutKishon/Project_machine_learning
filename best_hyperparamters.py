from utils import *
from classification import *



# searching for the best k for knn algo
def check_best_k_for_knn(features, labels):
    options_for_k = {k: 0 for k in range(1, 73, 2)}

    for _ in range(tests_num):
        x_train, x_test, y_train, y_test = split_data(
            features, labels)

        for k in range(1, 73, 2):
            acc_score = check_knn(x_train, y_train, x_test, y_test, k)
            options_for_k[k] += acc_score
    max_key = max(options_for_k, key=options_for_k.get)

    return max_key


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


def best_max_depth_for_decision_tree(features, labels):
    x_train, x_test, y_train, y_test = split_data(features, labels)

    train_accuracy = check_decision_tree(
        x_train, y_train, x_train, y_train, None)
    test_accuracy = check_decision_tree(x_train, y_train, x_test, y_test, None)
    print(
        f"max_depth = None : train_accuracy: {train_accuracy} , test accuracy: {test_accuracy}")

    train_accuracy = check_decision_tree(x_train, y_train, x_train, y_train, 4)
    test_accuracy = check_decision_tree(x_train, y_train, x_test, y_test, 4)
    print(
        f"max_depth = 4 : train_accuracy: {train_accuracy} , test accuracy: {test_accuracy}")


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


if __name__ == "__main__":

    dh = DataHolder()

    for label in ["hypertension", "heart_disease", "stroke"]:

        features, labels = get_features_labels(dh, label)
        # features = select_features(features, labels)

        print(f"\nall following results are for {label}")
        best_max_depth_for_decision_tree(features, labels)

        k = check_best_k_for_knn(features, labels)
        print(f"best k for knn algo is: {k}")

        criterion = check_best_criterion_for_decision_tree(features, labels)
        print(f"best criterion for decision_tree algo is: {criterion}")

        option_nb = best_option_for_naive_bayes(features, labels)
        print(f"best option for naive_bayes algo is: {option_nb}")
