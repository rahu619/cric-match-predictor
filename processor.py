# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:08:02 2022

@author: Dev_Me
"""
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

        # Retaining the predictor X variables alone in the dataframe
        self.dataframe.drop(
            columns=[column for column in self.dataframe if column not in self.predictors_arr], inplace=True)

    def encode_categories(self):
        """
        Encoding the predictor X and target Y variables as string categories cant be fed into a model directly
        """
        LE = LabelEncoder()
        for column in self.dataframe:
            self.dataframe[column] = LE.fit_transform(
                self.dataframe[column])

    def print_data_quality(self):
        print("--------------- Validating if there are NULL values ----------------\n")
        print(self.dataframe.info())
        print('--------------- Dataframe First 5 rows ----------------\n')
        print(self.dataframe.head())
        print()
        print('--------------- Validating Multivariate outliers in n-dimensional space ----------------\ \n')
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
