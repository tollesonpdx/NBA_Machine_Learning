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

def timestamp(start):
    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - start), 60)
    print(f'Program runtime: {int(minutes)} min {(seconds)} sec.')

def prep_data(source, split):

  lenTraining=int(len(source) * split)  
  print(source)
  denomanoms = source.abs().max()
  denomanoms[-1], denomanoms[-2]=1., 1.
  source/=denomanoms
  source.loc[source['nba_gms_plyed']>=240, 'success']=1
  source.loc[source['nba_gms_plyed'] <240, 'success']=0
  source=shuffle(source.drop(columns=['nba_gms_plyed']))  
  trainingData = source[:lenTraining]
  testingData = source[lenTraining:]
  return trainingData, testingData

def main():
  timeStart = time.time()
  split=.8
  toms_data = pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))
  trainData, testData = prep_data(toms_data, split)
  timestamp(timeStart)

main()
