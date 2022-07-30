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

    datamgr_obj = DataManager()
    df = datamgr_obj.read_csv('all_season_summary_with_percentage.csv')

    # The following fn can be used to include home/away win % in the default dataset
    # dataImporterObj.include_team_win_percentages(df)

    print(df)

    X_variables = ['home_team', 'away_team', 'toss_won', 'decision', 'referee', 'reserve_umpire', 'home_key_batsman',
                   'home_key_bowler', 'away_key_batsman', 'away_key_bowler', 'home_captain', 'away_captain', 'venue_name',
                   'home_playx1', 'away_playx1', 'home_team_win_percentage', 'away_team_win_percentage']
    y_variables = 'winner'

    processor_obj = PreProcessor(df, X_variables, y_variables)
    processor_obj.filter()
    processor_obj.split_and_expand(['home_playx1', 'away_playx1',
                                    'home_key_batsman', 'home_key_bowler', 'away_key_batsman', 'away_key_bowler'])
    processor_obj.encode_categories()
    processor_obj.print_data_quality()
    processor_obj.plotting_data_correlation()

    # TODO: Find and remove outliers from dataset (?)

    # retrieving processed dataframe
    df = processor_obj.dataframe

    # refined_X_variables = ['home_team', 'away_team', 'toss_won', 'decision', 'referee', 'reserve_umpire',
    #                        'home_key_batsm_1', 'home_key_batsm_2',
    #                        'home_key_bowl_1', 'home_key_bowl_2',
    #                        'away_key_batsm_1', 'away_key_batsm_2',
    #                        'away_key_bowl_1', 'away_key_bowl_2',
    #                        'home_captain', 'away_captain', 'venue_name',
    #                        'home_play_1', 'home_play_2', 'home_play_3', 'home_play_4', 'home_play_5', 'home_play_6', 'home_play_7', 'home_play_8', 'home_play_9', 'home_play_10', 'home_play_11',
    #                        'away_play_1', 'away_play_2', 'away_play_3', 'away_play_4', 'away_play_5', 'away_play_6', 'away_play_7', 'away_play_8', 'away_play_9', 'away_play_10', 'away_play_11',
    #                        'home_team_win_percentage', 'away_team_win_percentage']

    X = df[processor_obj.refined_X]
    y = df[y_variables]
    algorithm_obj = Algorithms(X, y)

    # Trying out RF model for our regression task
    random_forest_accuracy = algorithm_obj.random_forest_classifier()
    print("Random Forest Classifier model accuracy:", random_forest_accuracy)

    # TODO: Try out Hyper-parameter tuning to improve the score from ~55%


if __name__ == '__main__':
    main()
