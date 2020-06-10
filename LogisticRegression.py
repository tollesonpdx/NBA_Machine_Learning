import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# no_nans=[
#     'draft_pick', 'hght_noshoes','hght_wtshoes','wingspan','standing_reach',
#     'weight', 'body_fat', 'clg_games_plyd', 'pts_ppg', 'rpg,ast', 'fg2_pct', 
#     'fg3_pct', 'ft_pct', 'guards', 'forwards', 'centers', 'drafted', 'nba_gms_plyed']

def timestamp(start):
    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - start), 60)
    print(f'Program runtime: {int(minutes)} min {(seconds)} sec.')


def prep_data(source, split):

  print(source.isna().any())
  exit()
  print(source['nba_gms_plyed'].head())
  # denomanoms = source.abs().max()
  # source = source/denomanoms
  # print(source)
  trainingData = source[:lenTraining]
  testingData = source[lenTraining:]
  return trainingData, testingData

def main():
  timeStart = time.time()
  split=.8
  toms_data=pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))
  trainData, testData = prep_data(toms_data, .8)
  timestamp(timeStart)

main()
