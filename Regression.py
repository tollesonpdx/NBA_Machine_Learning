# -*- coding: utf-8 -*-
import os
import sys
import time
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
from sklearn.metrics import mean_squared_error, r2_score

plot_colors=['slategray', 'gold', 'navy', 'black', 'crimson', 'chocolate', 'y', 'mediumspringgreen', 'rebeccapurple', 'coral', 'olive', 'papayawhip', 'lightseagreen', 'brown', 'orange', 'khaki', 'pink', 'purple', 'bisque','red', 'tomato', 'turquoise', 'forestgreen', 'blue', 'cyan']
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def timestamp(start):
    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - start), 60)
    print(f'Program runtime: {int(minutes)} min {(seconds)} sec.')

def plot_confmat(cm):

    df_cm = pd.DataFrame(cm, range(2), range(2))
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.title(f'Baller or No Baller?: | test set confusion matrix')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

def prep_data(source, split):

  source=source.iloc[:,:-1]
  lenTraining=int(len(source) * split)  
  denomanoms = source.abs().max()
  denomanoms[-1], denomanoms[-2]=1., 1.
  source/=denomanoms
  source.loc[source['nba_gms_plyed']>=200, 'success']=1
  source.loc[source['nba_gms_plyed'] <200, 'success']=0
  source=shuffle(source.drop(columns=['nba_gms_plyed'])) 
  if 0:
    source=source.drop(columns=['draft_pick', 'drafted']) 
  trainingData=source[:lenTraining]
  testingData =source[lenTraining:]

  return trainingData, testingData

def regression(trainData, testData):
  
  confmat=np.zeros((2,2))
  
  X_train=trainData.drop(columns=['success']) 
  X_test =testData.drop(columns=['success']) 
  y_train=trainData['success'].values
  y_test =testData ['success'].values
  
  lm = LogisticRegression(random_state = 0)
  lm.fit(X_train, y_train)
  pred=lm.predict(X_test)
  correct=0
  for i in range(len(pred)):
    if y_test[i]==pred[i]:
      correct+=1
    confmat[int(pred[i])][int(y_test[i])]+=1

  # print(f'Percent correctly predicted by logistic regression model: {round(correct/len(y_test), 2)}%')
  # print('Coefficients: \n', lm.coef_)
  # print(f'Mean squared error: {round(mean_squared_error(y_test, pred), 2)}')
  # # The coefficient of determination: 1 is perfect prediction
  # print('Coefficient of determination: %.2f'
  #       % r2_score(y_test, pred))

  if 0:
    feature_importance=abs(lm.coef_[0])
    feature_importance=100.0*(feature_importance/feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos=np.arange(sorted_idx.shape[0]) + .5

    featfig=plt.figure()
    featax=featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center',color=plot_colors, edgecolor='black')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(X_train.columns)[sorted_idx], fontsize=12)
    featax.set_xlabel('Relative Feature Importance For NBA Success', fontsize=14)

    plt.tight_layout()   
    plt.show()

  return correct/len(y_test), confmat

def main():
  timeStart = time.time()
  split=.8
  toms_data = pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))
  scores=np.zeros(100)
  
  for i in range(100):
    trainData,testData=prep_data(toms_data, split)
    scores[i],confmat=regression(trainData, testData)
  # print(scores)
  TP,FP,FN,TN=confmat[0][0],confmat[0][1],confmat[1][0],confmat[1][1]
  accuracy =(TP+TN)/(TP+FP+FN+TN)
  precision=TP/(TP+FP)
  recall   =TP/(TP+FN)  
  print(f'accuracy  = {accuracy} ')
  print(f'precision = {precision}')
  print(f'recall    = {recall}   ')
  
  img =plt.imread("court.jpg")
  plt.xlabel('Trial', fontsize=16)
  plt.ylabel('Percent of Correct Predictions', fontsize=16)
  plt.title('Overall Predictive Percent: Regression', fontsize=16)
  plt.imshow(img, zorder=0, extent=[-5,100, 30,100])
  avg=sum(scores*100)/len(scores)
  avgs=[avg for i in range(len(scores))]
  red_patch = mpatches.Patch(color='red', label='Avg: {:6.2f}%'.format(avg))
  black_patch = mpatches.Patch(color='black', label='Individual Trial')
  plt.legend(handles=[red_patch, black_patch])
  plt.scatter(np.arange(len(scores)), avgs, s=4, c='red', zorder=1)
  plt.scatter(np.arange(len(scores)), scores*100, c='black', zorder=1)
  plt.show()
  plot_confmat(confmat)

  timestamp(timeStart)

main()
