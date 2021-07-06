from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from utils import *




def split_data(data, labels):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data and it's labels columns
    :return: train set, test set , labels of training and test sets data frames
    """
    print(len(data))
    print(len(labels))
    return train_test_split(data, labels, test_size=0.5)




def knn(x_train, y_train, x_test, n_neighbors=71):

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Train the model using the training sets
    model.fit(x_train, y_train)

    return model.predict(x_test)


# def check_accuracy():
if __name__ == "__main__":
    avgs_dict = {}
    dh = DataHolder()
    features = dh.get_features("hypertension")
    labels = dh.get_labels("hypertension")
    splits_num = 100
    for _ in range (splits_num):
        X_train, X_test, y_train, y_test = split_data(features, labels)
        for i in range(1, 73):
            y_pred = knn(X_train, y_train, X_test, i)
            acc_score = metrics.accuracy_score(y_test, y_pred)
            print(f"Accuracy for {i} neighbours:", acc_score)
            avgs_dict[i] = avgs_dict.get(i, 0) + acc_score
    max_key = max(avgs_dict, key=avgs_dict.get)
    max_val = max(avgs_dict.values())
    print(f"best accuracy is for {max_key} neighbours: {max_val/splits_num}")


# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# 5110 samples (in knn: k = 71)
