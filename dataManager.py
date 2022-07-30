# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 08:09:17 2022

@author: Dev_Me
"""
import os
import pandas as pd
from varname import nameof


class DataManager:
    DATA_DIR = 'data'
    WIN_PERCENTAGE_INCLUDED_FILENAME = 'all_season_summary_with_percentage.csv'
    home_team_win_percentage = away_team_win_percentage = 0

    def read_csv(self, file_name):
        return self.__read_file(file_name, pd.read_csv)

    def read_excel(self, file_name):
        return self.__read_file(file_name, pd.read_excel)

    def include_team_win_percentages(self, df):
        return self.__calculate_team_win_percentages(df)

    def __calculate_team_win_percentages(self, df, save_to_file=True):
        df[nameof(self.home_team_win_percentage)] = 0.0
        df[nameof(self.away_team_win_percentage)] = 0.0

        for index, row in df.iterrows():
            filtered_df = df[
                (df.home_team == row.home_team)
                & (df.away_team == row.away_team)
                & (pd.to_datetime(df.start_date) < pd.to_datetime(row.start_date))
                #  & (df.venue_id == row.venue_id)
            ]

            if len(filtered_df) > 0:
                home_team_wins = len(
                    filtered_df[filtered_df.winner == filtered_df.home_team])
                away_team_wins = len(
                    filtered_df[filtered_df.winner == filtered_df.away_team])
                total_matches = len(filtered_df)

                df.at[index,  nameof(self.home_team_win_percentage)] = (
                    home_team_wins / total_matches) * 100
                df.at[index,  nameof(self.away_team_win_percentage)] = (
                    away_team_wins / total_matches) * 100

        if save_to_file:
            df.to_csv(os.path.join(self.DATA_DIR,
                      self.WIN_PERCENTAGE_INCLUDED_FILENAME), encoding='utf-8')

        return df

    def __read_file(self, file_name, func):
        path = os.path.join(self.DATA_DIR, file_name)
        print('File path:', path)
        try:
            return func(path)
        except IOError as e:
            print('Error reading file from path: %s' % (path))
            print(e)
            return None
