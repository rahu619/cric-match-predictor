# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:56:32 2022

@author: Dev_Me
"""
import numpy as np
import re
from dataImporter import DataImporter
from processor import PreProcessor


def main():
    print('importing dataset')
    dataImporterObj = DataImporter()
    df = dataImporterObj.read_csv('all_season_summary.csv', 'data')
    print(df)

    # TODO: Include players in dataset
    X = ['home_team', 'away_team', 'toss_won', 'decision', 'referee', 'reserve_umpire', 'home_key_batsman',
         'home_key_bowler', 'away_key_batsman', 'away_key_bowler', 'home_captain', 'away_captain', 'venue_name']
    y = 'winner'

    preProcessor = PreProcessor(df, X, y)
    preProcessor.filter()
    preProcessor.encode_categories()
    preProcessor.print_data_quality()

    # TODO: Find outliers from dataset


if __name__ == '__main__':
    main()
