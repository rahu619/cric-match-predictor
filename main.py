# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:56:32 2022

@author: Dev_Me
"""
from dataImporter import DataImporter
from processor import PreProcessor


def main():
    print('importing dataset')
    dataImporterObj = DataImporter()
    df = dataImporterObj.read_csv('all_season_summary.csv', 'data')
    print(df)

    X = ['home_team', 'away_team', 'toss_won', 'decision', 'referee', 'reserve_umpire', 'home_key_batsman',
         'home_key_bowler', 'away_key_batsman', 'away_key_bowler', 'home_captain', 'away_captain', 'venue_name',
         'home_playx1', 'away_playx1']
    y = 'winner'

    preProcessor = PreProcessor(df, X, y)
    preProcessor.filter()
    preProcessor.split_and_expand(['home_playx1', 'away_playx1'])
    preProcessor.encode_categories()
    preProcessor.print_data_quality()

    # TODO: Find outliers from dataset

    # Visualizers - use seaborn perhaps(?)
    # No:of times the home team won vs away team
    # Most wins per team
    # Impact of winning a toss

    # Calculate correlation of dependent variables
    # and visualize it (?)


if __name__ == '__main__':
    main()
