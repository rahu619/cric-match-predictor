# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 08:09:17 2022

@author: Dev_Me
"""
import os
import pandas as pd

class DataImporter:
    def read_csv(self, file_name, file_path=''):
        """
        @type  file_name: string
        @type  file_path: string
        @param file_name: The name of the file
        @param file_path: The optional directory path of the csv file.
        @return: the dataframe of the csv file.
        """
        return self.__read_file(os.path.join(file_path, file_name), pd.read_csv)

    def read_excel(self, file_name, file_path=''):
        return self.__read_file(os.path.join(file_path, file_name), pd.read_excel)
            
            
    def __read_file(self, path, func):
        print('File path:' , path)
        try:
            return func(path)
        except IOError as e:
            print('Error reading file from path: %s' % (path))
            print(e)
            return None
        
