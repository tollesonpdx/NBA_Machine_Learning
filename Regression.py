import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def timestamp(start):
    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - start), 60)
    print(f'Program runtime: {int(minutes)} min {(seconds)} sec.')

def prep_data(source, split):

  lenTraining=int(len(source) * split)  
  denomanoms = source.abs().max()
  denomanoms[-1], denomanoms[-2]=1., 1.
  source/=denomanoms
  source.loc[source['nba_gms_plyed']>=240, 'success']=1
  source.loc[source['nba_gms_plyed'] <240, 'success']=0
  source=shuffle(source.drop(columns=['nba_gms_plyed']))  
  trainingData=source[:lenTraining]
  testingData =source[lenTraining:]

  return trainingData, testingData

def regression(trainData, testData):

  X_train=trainData.drop(columns=['success']) 
  X_test =testData.drop(columns=['success']) 
  y_train=trainData['success'].values
  y_test =testData ['success'].values
  
  lm=LogisticRegression().fit(X_train, y_train)
  pred=lm.predict(X_test)
  correct=0
  for i in range(len(pred)):
    if y_test[i]==pred[i]:
      correct+=1
  print(f'Percent correctly predicted by Logistic regression: {correct/len(y_test)}')

def main():
  timeStart = time.time()
  split=.8
  toms_data = pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))
  trainData, testData = prep_data(toms_data, split)
  regression(trainData, testData)
  
  timestamp(timeStart)

main()
