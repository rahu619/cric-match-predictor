# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 09:08:02 2022

@author: Dev_Me
"""
from sklearn.preprocessing import LabelEncoder

class PreProcessor:    
    def __init__(self, df, predictors_arr, target):
        self.dataframe = df
        self.predictors_arr = predictors_arr
        self.target = target
        
        #Performing Exploratory Data Analysis
        print('Analysing shape')
        print(df.shape)
         
        
    def remove_null(self):
        #The predictor X variable subset array will be preprocessed
        self.df.dropna(subset=self.predictors_arr, inplace=True)
        print(self.df.shape)
            
          
    def encode_categories(self):
        """
        Encoding the predictor X and target Y variables as string categories cant be fed into a model directly
        """
        LE = LabelEncoder()        
        for predictor in self.predictors_arr:
            self.dataframe[predictor] = LE.fit_transform(self.dataframe[predictor])
        
        self.dataframe[self.target] = LE.fit_transform(self.dataframe[self.target])
        

    def print_data_quality(self):
        print("--------------- Validating if there are NULL values ----------------\n")
        print(self.df.info()) 
        print('--------------- Dataframe First 5 rows ----------------\n')
        print(self.df.head())
        print()        
        print('--------------- Validating Multivariate outliers in n-dimensional space ----------------\ \n')
        print(self.df.describe()) 
        print()

    #region Properties
    
    @property
    def dataframe(self):
        return self._df
    
    @dataframe.setter
    def dataframe(self, value):
        self._df = value
    
    #endregion