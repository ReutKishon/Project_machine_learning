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
    Residence_type = []
    smoking_status = []
    hypertension = []
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
        Residence_type.append(row[7])
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
    residence_type_encoded = le.fit_transform(Residence_type)
    smoking_status_encoded = le.fit_transform(smoking_status)

    features = list(zip(gender_encoded, age, heart_disease, married_encoded, work_type_encoded,
                    residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded))
    return features, labels


def Knn(X_train, y_train, X_test):

    model = KNeighborsClassifier(n_neighbors=71)

    # Train the model using the training sets
    model.fit(X_train, y_train)

    return model.predict(X_test)


# def check_accuracy():
if __name__ == "__main__":
    features, labels = read_data_from_file()
    X_train, X_test, y_train, y_test = split_data(features, labels)

    y_pred = Knn(X_train, y_train, X_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# 5110 samples (in knn: k = 71)
