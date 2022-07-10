# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:08:02 2022

@author: Dev_Me
"""
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder


class PreProcessor:
    def __init__(self, df, predictors_arr, target):
        self.dataframe = df
        self.predictors_arr = predictors_arr
        self.target = target

        # Performing Exploratory Data Analysis
        print('Analysing shape')
        print(df.shape)

    def filter(self):
        # Removing null ones
        self.dataframe.dropna(subset=self.predictors_arr, inplace=True)
        print(self.dataframe.shape)

        # Checking for duplicates
        self.dataframe[self.dataframe.duplicated(keep=False)]

        # Retaining the predictor X variables alone in the dataframe
        self.dataframe.drop(
            columns=[column for column in self.dataframe if column not in self.predictors_arr], inplace=True)

    def split_and_expand(self, labels_arr):
        # splitting and expanding specific labels
        for label in labels_arr:
            temp = self.dataframe[label].str.split(',', expand=True).applymap(
                lambda x: np.nan if x is None else x)
            new_column = label[0:len(label)-2]
            self.dataframe[[new_column+'_' +
                            str(i) for i in range(1, temp.shape[1] + 1)]] = temp

        # removing labels_arr
        self.dataframe.drop(columns=labels_arr)

    def encode_categories(self, encode_labels_arr=None):
        """
        To encoding the categorical variables into numerical labels
        """
        # TODO: Introduce rescaling by using MinMaxScaler(?)
        LE = LabelEncoder()
        for column in (self.dataframe[encode_labels_arr] if encode_labels_arr != None else self.dataframe):
            self.dataframe[column] = LE.fit_transform(
                self.dataframe[column])

    def print_data_quality(self):
        print("--------------- Validating if there are NULL values ----------------\n")
        print(self.dataframe.info())
        print('--------------- Displaying First 5 rows ----------------\n')
        print(self.dataframe.head())
        print()
        print('--------------- Validating Multivariate outliers in n-dimensional space ----------------\n')
        print(self.dataframe.describe())
        print()

    # region Properties

    @property
    def dataframe(self):
        return self._df

    @dataframe.setter
    def dataframe(self, value):
        self._df = value

    # endregion
