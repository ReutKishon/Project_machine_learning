import codecs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper
import numpy as np
from sklearn.impute import KNNImputer


def read_data_from_file():
    """
    read data from healthcare-dataset-stroke-data csv file
    :return: DataFrame of the data after transformation
    """
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    le = LabelEncoder()
    # array of catagorical fields
    clean_ups = {"smoking_status": {"smokes": 3, "never smoked": 1, "formerly smoked": 2, "Unknown": None}}
    df = df.replace(clean_ups)
    not_num_cols = ["gender", "ever_married",
                    "work_type", "Residence_type"]

    df = df.drop('id', axis=1)
    impute = KNNImputer(n_neighbors=5, weights='uniform')
    df['bmi'] = impute.fit_transform(df[['bmi']])
    df['smoking_status'] = impute.fit_transform(df[['smoking_status']])

    # Generates a feature definition list which can be passed into DataFrameMapper
    categorical_feature = gen_features(columns=not_num_cols,
                                       classes=[LabelEncoder])
    # map of encoded categorical feature
    mapper = DataFrameMapper(features=categorical_feature)

    tmp_df = pd.DataFrame(mapper.fit_transform(df), columns=not_num_cols)
   
    for col in not_num_cols:
        df[col] = tmp_df[col].values
    print(df)
    exit()
    return df


def split_data(data, labels):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data and it's labels columns
    :return: train set, test set , labels of training and test sets data frames
    """
    return train_test_split(data, labels, test_size=0.5)


def get_features_labels(dh, label):
    features = dh.get_features(label)
    labels = dh.get_labels(label)
    return features, labels


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataHolder(metaclass=Singleton):
    def __init__(self):
        self.df = read_data_from_file()

    def get_labels(self, name):
        return self.df.loc[:, name]

    def get_features(self, label_name):
        return self.df.loc[:, self.df.columns != label_name]
