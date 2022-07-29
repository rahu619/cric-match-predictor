# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:56:32 2022

@author: Dev_Me
"""
from dataManager import DataManager
from processor import PreProcessor
from algorithms import Algorithms


def main():
    print('importing dataset')

    dataImporterObj = DataManager()
    df = dataImporterObj.read_csv('all_season_summary.csv')

    # The following fn can be used to include home/away win % in the default dataset
    dataImporterObj.include_team_win_percentages(df)

    print(df)

    X_variables = ['home_team', 'away_team', 'toss_won', 'decision', 'referee', 'reserve_umpire', 'home_key_batsman',
                   'home_key_bowler', 'away_key_batsman', 'away_key_bowler', 'home_captain', 'away_captain', 'venue_name',
                   'home_playx1', 'away_playx1', 'home_team_win_percentage', 'away_team_win_percentage']
    y_variables = 'winner'

    preProcessorObj = PreProcessor(df, X_variables, y_variables)
    preProcessorObj.filter()
    preProcessorObj.split_and_expand(
        ['home_playx1', 'away_playx1',
         'home_key_batsman', 'home_key_bowler', 'away_key_batsman', 'away_key_bowler']
    )
    preProcessorObj.encode_categories()
    preProcessorObj.print_data_quality()
    preProcessorObj.plotting_data_correlation()
    # TODO: Find and remove outliers from dataset (?)

    # retrieving processed dataframe
    df = preProcessorObj.dataframe

    X = df[X_variables]
    y = df[y_variables]
    algorthimObj = Algorithms(X, y)

    # Trying out RF model for our regression task
    random_forest_accuracy = algorthimObj.random_forest_classifier()
    print("Random Forest Classifier model accuracy:", random_forest_accuracy)

    # TODO: Try out Hyper-parameter tuning to improve the score from ~55%


if __name__ == '__main__':
    main()
