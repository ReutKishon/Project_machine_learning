from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2


def select_features(features, labels):
    """
    Select features according to the k highest scores.
    :param features: dataframe of fearues
    :param labels: dataframe of labels
    :return: dataframe with the important features only
    """
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(features, labels)
    per = [ ]
    # for loop to calculate variable importance percentage
    for i in fs.scores_:
        per.append(round(((i / sum(fs.scores_)) * 100), 3))

    # Creating a new dataframe to display the Chi-square and Importance(%) scores
    features_data = pd.DataFrame({'Feature': features.columns, 'Scores': fs.scores_,
                                  'Importance(%)': per}).sort_values(by=[ 'Scores' ], ascending=False)


    # Creating an insignificant variable to store the variables with Importance % < 0.005
    insignificant = features_data.loc [ features_data [ 'Importance(%)' ] < 0.005 ] [ 'Feature' ].unique()
    features = features.drop(insignificant, axis =1)
    return norm_data(features)

def norm_data(df):
    columns_names = df.columns
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=columns_names)

def read_data_from_file():
    """
    read data from healthcare-dataset-stroke-data csv file
    :return: DataFrame of the data after transformation
    """
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    # array of catagorical fields
    clean_ups = {"smoking_status": {"smokes": 3, "never smoked": 1, "formerly smoked": 2, "Unknown": None},
                 "gender": {"Male": 0, "Female": 1, "Other": 0.5}}
    df = df.replace(clean_ups)
    not_num_cols = ["ever_married", "work_type", "Residence_type"]

    df = df.drop('id', axis=1)

    bmi_impute = KNNImputer(n_neighbors=71, weights='uniform') #71 is the sqrt of the data rows length
    smoking_impute = KNNImputer(n_neighbors=len(df['smoking_status'])/3, weights='distance') #3 smoking classes

    df['bmi'] = bmi_impute.fit_transform(df[['bmi']])
    df['smoking_status'] = smoking_impute.fit_transform(df[['smoking_status']])
    # Generates a feature definition list which can be passed into DataFrameMapper
    categorical_feature = gen_features(columns=not_num_cols,
                                       classes=[LabelEncoder])
    # map of encoded categorical feature
    mapper = DataFrameMapper(features=categorical_feature)

    tmp_df = pd.DataFrame(mapper.fit_transform(df), columns=not_num_cols)

    for col in not_num_cols:
        df[col] = tmp_df[col].values
    treat_outliers(df)

    #normalize data
    return df


def treat_outliers(df):
    """
    by plotting the values of bmi and avg glucose we found that there are outliers in both
    the way to handle it is by comparing the outliers to mean +3 std and check the values
    :param df: datefraem of the healthcare info
    :return: df
    """

    plt.subplot(1, 2, 1)
    sns.distplot(df [ 'bmi' ])
    plt.subplot(1, 2, 2)
    df [ 'bmi' ].plot.box(figsize=(16, 5))
    plt.show()
    bmi_high1 = np.percentile(df [ 'bmi' ], 50)
    df[ 'bmi' ] = np.where(df[ 'bmi' ] > bmi_high1, bmi_high1, df [ 'bmi' ])

    #since scaning the data of bmi let us know that there are still too many outliers, we repeat the procedure only for it
    bmi_high1 = np.percentile(df [ 'bmi' ], 99.7)
    df[ 'bmi' ] = np.where(df[ 'bmi' ] > bmi_high1, bmi_high1, df [ 'bmi' ])



def split_data(data, labels, test_size=0.25):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data and it's labels columns
    :return: train set, test set , labels of training and test sets data frames
    """
    return train_test_split(data, labels, test_size=test_size)


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
