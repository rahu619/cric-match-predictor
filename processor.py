# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:08:02 2022

@author: Dev_Me
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


class PreProcessor:
    """Processes the input variables"""

    def __init__(self, df, predictors_arr, target):
        self.dataframe = df
        self.predictors_arr = predictors_arr
        self.target = target
        self.variables = np.append(self.predictors_arr, self.target)
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

            # concatenating the new columns into our X variable set
            self.predictors_arr.extend(split_columns)

        # removing irrelevant X variables from the dataframe
        self.dataframe.drop(labels_arr, axis=1, inplace=True)

        # modifying X variables to accomodate new variables
        self.predictors_arr = np.setdiff1d(self.predictors_arr, labels_arr)

    def encode_categories(self, encode_labels_arr=None):
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
