# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:08:02 2022

@author: Dev_Me
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum


class Scaler(Enum):
    MINMAX = 1,
    STANDARD = 2,
    MAXABS = 3


class PreProcessor:
    """Processes the input variables"""

    def __init__(self, df, predictors_arr, target):
        self.dataframe = df
        self.predictors_arr = predictors_arr
        self.target = target
        self.variables = np.append(self.predictors_arr, self.target)
        self.BRACKET_CHAR = '('
        # Performing Exploratory Data Analysis
        print('Analysing shape')
        print(df.shape)

    def filter(self):
        # Removing null ones
        self.dataframe.dropna(subset=self.predictors_arr, inplace=True)
        print(self.dataframe.shape)

        # Checking for duplicates
        self.dataframe[self.dataframe.duplicated(keep=False)]

        # Retaining the predictor X, target y variables alone in the dataframe
        self.dataframe.drop(
            columns=[column for column in self.dataframe if column not in self.variables], inplace=True)

    def split_and_expand(self, labels_arr):
        """ Splits and expands the specified labels """
        for label in labels_arr:
            temp = self.dataframe[label].str.split(',', expand=True).applymap(
                lambda x: np.nan if x is None else x)
            split_columns = [label + '_' + str(i)
                             for i in range(1, temp.shape[1] + 1)]
            self.dataframe[split_columns] = temp

            for column in self.dataframe[split_columns]:
                self.dataframe[column] = self.dataframe[column].map(
                    lambda x: self.__cleanup_brackets(x))

            # concatenating the new columns into our X variable set
            self.predictors_arr.extend(split_columns)

        # removing irrelevant X variables from the dataframe
        self.dataframe.drop(labels_arr, axis=1, inplace=True)

        # modifying X variables to accomodate new variables
        self.predictors_arr = np.setdiff1d(self.predictors_arr, labels_arr)

    def __get_encode_group_of_columns_dict(self, columns_arr):
        encoded_unique_dict = {}
        unique_values = []
        for column in self.dataframe[columns_arr]:
            for item in self.dataframe[column].values:
                if item not in unique_values:
                    unique_values.append(item)

        LE = LabelEncoder()
        for i, value in enumerate(LE.fit_transform(unique_values)):
            encoded_unique_dict[unique_values[i]] = value

        return encoded_unique_dict

    # As most players have an invalid bracket character appended to the name
    def __cleanup_brackets(self, value):
        if self.BRACKET_CHAR in value:
            return value[0:value.index(self.BRACKET_CHAR) - 1]
        return value

    def encode_group_of_columns(self, arr_columns_to_modify_arr, unique_map_columns_arr):
        encoded_unique_dict = self.__get_encode_group_of_columns_dict(
            unique_map_columns_arr)
        # print('encoded_unique_dict: ', encoded_unique_dict)
        for columns_arr in arr_columns_to_modify_arr:
            for column in self.dataframe[columns_arr]:
                self.dataframe[column].replace(
                    encoded_unique_dict, inplace=True)
            # print('columns_arr: ', self.dataframe[columns_arr])

    def encode_columns(self, encode_labels_arr=None):
        """To encoding the categorical variables into numerical labels"""
        # TODO: Introduce rescaling by using MinMaxScaler(?)
        LE = LabelEncoder()
        for column in (self.dataframe[encode_labels_arr] if encode_labels_arr is not None else self.dataframe):
            self.dataframe[column] = LE.fit_transform(self.dataframe[column])

    def print_data_quality(self):
        """Prints the dataframe meta info"""
        print("--------------- Validating if there are NULL values ----------------\n")
        print(self.dataframe.info())
        print('--------------- Displaying First 5 rows ----------------\n')
        print(self.dataframe.head())
        print()
        print('--------------- Validating Multivariate outliers in n-dimensional space ----------------\n')
        print(self.dataframe.describe())
        print()

    def plotting_data_correlation(self):
        """ For calculating and visualizing the correlations between dependent and independent variables. This approach could assert our assumptions for selecting appropriate predictor variables """
        corr_matrix = self.dataframe.corr()
        print(corr_matrix[self.target].sort_values(ascending=False))
        sns.heatmap(corr_matrix,
                    xticklabels=self.dataframe.columns,
                    yticklabels=self.dataframe.columns)
        # plt.show()

    def scale(self, scaler, arr):
        if scaler == Scaler.MINMAX:
            return MinMaxScaler().fit_transform(arr)
        elif scaler == Scaler.STANDARD:
            return StandardScaler().fit_transform(arr)
        elif scaler == Scaler.MAXABS:
            return MaxAbsScaler().fit_transform(arr)

    def df_write(self):
        self.dataframe.to_csv('data\output.csv', encoding='utf-8', index=False)

    # region Properties

    @property
    def dataframe(self):
        """Getter for the cricket dataframe"""
        return self._df

    @property
    def refined_X(self):
        """Getter for the newly refined X variables"""
        return self.predictors_arr

    @dataframe.setter
    def dataframe(self, value):
        self._df = value

    # endregion
