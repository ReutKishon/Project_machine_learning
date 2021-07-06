from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from csv import reader


def read_data_from_file():
    """
    read data from healthcare-dataset-stroke-data csv file
    :return: 1) listof lists (features) (after encoding the categorical fields)
             2) list of integers (labels of samples)
    """

    # skip first line i.e. read header first and then iterate over each row od csv as a list
    with open('healthcare-dataset-stroke-data.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            return encoded_data(csv_reader)


def split_data(data, labels):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data and it's labels columns
    :return: train set, test set , labels of training and test sets data frames
    """
    return train_test_split(data, labels, test_size=0.5)


def encoded_data(csv_reader):
    """
    label encoding - represents categorical columns in a numerical column (map each value to a number),
    using Label Encoder in sklearn.
    :param data: A reader object which will iterate over lines in the given csvfile.
    :return: 1) list of lists s.t each cell(list) represents sample from the csvfile.
             2) list of integers represents the labels of each sample 
                (the label is 1 if the patient has hypertension , 0 otherwise)
    """         
    gender = []
    age = []
    ever_married = []
    work_type = []
    residence_type = []
    smoking_status = []
    heart_disease = []
    avg_glucose_level = []
    bmi = []
    labels = []

    # appending data to lists respectively
    for row in csv_reader:

        gender.append(row[1])
        age.append(float(row[2]))
        heart_disease.append(int(row[4]))
        ever_married.append(row[5])
        work_type.append(row[6])
        residence_type.append(row[7])
        avg_glucose_level.append(float(row[8]))
        bmi.append(-1) if (row[9] == 'N/A') else bmi.append(float(row[9]))
        smoking_status.append(row[10])
        labels.append(int(row[3]))

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers.
    gender_encoded = le.fit_transform(gender)
    married_encoded = le.fit_transform(ever_married)
    work_type_encoded = le.fit_transform(work_type)
    residence_type_encoded = le.fit_transform(residence_type)
    smoking_status_encoded = le.fit_transform(smoking_status)

    features = list(zip(gender_encoded, age, heart_disease, married_encoded, work_type_encoded,
                    residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded))
    return features, labels


def knn(x_train, y_train, x_test, n_neighbors=71):

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model using the training sets
    model.fit(x_train, y_train)

    return model.predict(x_test)


# def check_accuracy():
if __name__ == "__main__":
    avgs_dict = {}
    features, labels = read_data_from_file()
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
