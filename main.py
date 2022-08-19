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

    # print(df)

    X_variables = ['home_team', 'away_team', 'toss_won', 'decision', 'reserve_umpire', 'home_key_batsman',
                   'home_key_bowler', 'away_key_batsman', 'away_key_bowler', 'home_captain', 'away_captain', 'venue_name',
                   'home_playx1', 'away_playx1', 'away_team_win_percentage']
    #    'referee', 'home_team_win_percentage', ]
    y_variables = 'winner'

    processor_obj = PreProcessor(df, X_variables, y_variables)
    processor_obj.filter()

    # Modifies list of features to include the split columns
    processor_obj.split_and_expand(['home_playx1', 'away_playx1',
                                    'home_key_batsman', 'home_key_bowler', 'away_key_batsman', 'away_key_bowler'])

    unique_map_team_arr = ['home_team', 'away_team']
    processor_obj.encode_group_of_columns(
        [unique_map_team_arr], unique_map_team_arr)

    unique_map_players_arr = [
        'home_playx1_1', 'home_playx1_2', 'home_playx1_3', 'home_playx1_4', 'home_playx1_5', 'home_playx1_6', 'home_playx1_7', 'home_playx1_8', 'home_playx1_9', 'home_playx1_10', 'home_playx1_11',
        'away_playx1_1', 'away_playx1_2', 'away_playx1_3', 'away_playx1_4', 'away_playx1_5', 'away_playx1_6', 'away_playx1_7', 'away_playx1_8', 'away_playx1_9', 'away_playx1_10', 'away_playx1_11',
    ]
    processor_obj.encode_group_of_columns([
        unique_map_players_arr,
        [
            'home_key_batsman_1', 'home_key_batsman_2',
            'away_key_batsman_1', 'away_key_batsman_2',
            'home_key_bowler_1', 'home_key_bowler_2',
            'away_key_bowler_1', 'away_key_bowler_2',
        ],
        [
            'home_captain', 'away_captain'
        ]
    ], unique_map_players_arr)

    processor_obj.encode_columns(
        ['toss_won', 'decision', 'reserve_umpire', 'venue_name', 'away_team_win_percentage', 'winner'])
    processor_obj.print_data_quality()
    processor_obj.plotting_data_correlation()

    # TODO: Find and remove outliers from dataset (?)

    # retrieving processed dataframe
    df = processor_obj.dataframe

    X = df[processor_obj.refined_X]
    y = df[y_variables]
    algorithm_obj = Algorithms(X, y)

    # Trying out RF model for our regression task
    random_forest_classfier_accuracy = algorithm_obj.random_forest_classifier()
    print("Random Forest Classifier model accuracy:",
          random_forest_classfier_accuracy)

    # support_vector_machine_accuracy = algorithm_obj.support_vector_machine()
    # print("SVM model accuracy:", support_vector_machine_accuracy)

    # TODO: Try out Hyper-parameter tuning to improve the score from ~51%


if __name__ == '__main__':
    main()
