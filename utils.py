import codecs

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn_pandas import gen_features
from sklearn_pandas import DataFrameMapper
import numpy as np
from sklearn.impute import SimpleImputer

def read_data_from_file():
    """
    read data from healthcare-dataset-stroke-data csv file
    :return: DataFrame of the data after transformation
    """
    # skip first line i.e. read header first and then iterate over each row od csv as a list
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    le = LabelEncoder()
    not_num_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    feature_def = gen_features(columns=not_num_cols,
                               classes=[LabelEncoder])
    mapper5 = DataFrameMapper(feature_def)
    tmp_df = pd.DataFrame(mapper5.fit_transform(df), columns=not_num_cols)
    for col in not_num_cols:
        df[col] = tmp_df[col].values
    imp.fit(df)
    tmp_arr = imp.transform(df)
    df = pd.DataFrame(tmp_arr, columns=df.columns)
    return df


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DataHolder(metaclass=Singleton):
    def __init__(self):
        self.df = read_data_from_file()


    def get_labels(self, name):
        return self.df.loc[:, name]

    def get_features(self, label_name):
        return self.df.loc [ :, self.df.columns != label_name ]




