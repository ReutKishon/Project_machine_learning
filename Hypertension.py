from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from csv import reader


def read_data_from_file():
    """
    read data from healthcare-dataset-stroke-data csv file
    :return: listof tuples (features)
    """

    # skip first line i.e. read header first and then iterate over each row od csv as a list
    with open('healthcare-dataset-stroke-data.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            return encoded_data(csv_reader)


def encoded_data(csv_reader):
    """
     data from healthcare-dataset-stroke-data csv file
    :return: DictReader object
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

    for row in csv_reader:

        gender.append(row[1])
        age.append(float(row[2]))
        hypertension.append(int(row[3]))
        heart_disease.append(int(row[4]))
        ever_married.append(row[5])
        work_type.append(row[6])
        Residence_type.append(row[7])
        avg_glucose_level.append(float(row[8]))
        bmi.append(-1) if (row[9] == 'N/A') else bmi.append(float(row[9]))
        smoking_status.append(row[10])
        labels.append(int(row[11]))

    # creating labelEncoder
    le = preprocessing.LabelEncoder()

    # Converting string labels into numbers.
    gender_encoded = le.fit_transform(gender)
    married_encoded = le.fit_transform(ever_married)
    work_type_encoded = le.fit_transform(work_type)
    residence_type_encoded = le.fit_transform(Residence_type)
    smoking_status_encoded = le.fit_transform(smoking_status)

    features = list(zip(gender_encoded, age, hypertension, heart_disease, married_encoded, work_type_encoded,
                    residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded))
    return features, labels


def Knn(features, labels):
    model = KNeighborsClassifier(n_neighbors=71)

    # Train the model using the training sets
    model.fit(features, labels)

    # Predict Output    /// stopped here
    predicted = model.predict(
        (0, 39, 0, 0, 1, 0, 1, 75.80, 22))  # 0:Overcast, 2:Mild
    print(predicted)


if __name__ == "__main__":
    features, labels = read_data_from_file()
    Knn(features, labels)


# Challenges: five of the dataset fields are strings.
# In order to implemement knn algorithm we need to calculate Euclidian distance
# between some sample and it's neighbors. hence all of the fields should be numarical.
# 5110 samples (in knn: k = 71)
