import os
import sys
import time
import math
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def pdf(x,μ,𝜎):

    e_term = math.exp(-1*((x-μ)**2)/(2*(𝜎**2)))
    normal = 1/(math.sqrt(2*math.pi*𝜎))
    N = e_term * normal
    return N if N > 0 else sys.float_info.min

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def timestamp(start):
    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - start), 60)
    print(f'Program runtime: {int(minutes)} min {(seconds)} sec.')

def plot_confmat(cm):

    df_cm = pd.DataFrame(cm, range(2), range(2))
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
  source.loc[source['nba_gms_plyed']>=240, 'success']=1
  source.loc[source['nba_gms_plyed'] <240, 'success']=0
  source=shuffle(source.drop(columns=['nba_gms_plyed']))  
  trainingData=source[:lenTraining]
  testingData =source[lenTraining:]

  return trainingData, testingData

def main():
  timeStart = time.time()
  
  confmats=[]
  split=.6
  toms_data = pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))
  scores=np.zeros(100)

  for x in range(100):
    score=0
    trainData,testData   =prep_data(toms_data, split)
    train_baller_prior   =np.sum(trainData, axis=0)[-1]/len(trainData)
    train_no_baller_prior=(len(trainData)-np.sum(trainData, axis=0)[-1])/len(trainData)
    train_baller,train_no_baller,=trainData[trainData.iloc[:,-1].values==1],trainData[trainData.iloc[:,-1].values==0]
    train_baller_means   ,train_baller_stdvs   =np.mean(train_baller,    axis=0)[:-1],np.std(train_baller,    axis=0)[:-1]
    train_no_baller_means,train_no_baller_stdvs=np.mean(train_no_baller, axis=0)[:-1],np.std(train_no_baller, axis=0)[:-1]  
    train_baller_stdvs[train_baller_stdvs==0.0],train_no_baller_stdvs[train_no_baller_stdvs==0.0]=.000001,.000001
    confmat=np.zeros((2,2))
    #loop over the test data
    for i in range(len(testData)):
      baller=math.log  (train_baller_prior,   10)
      noballer=math.log(train_no_baller_prior,10)
      correct_class=testData.iloc[i][-1]
      for j in range(len(testData.iloc[i])-1):
        baller  +=math.log((pdf(testData.iloc[i][j],train_baller_means   [j],train_baller_stdvs   [j])), 10)
        noballer+=math.log((pdf(testData.iloc[i][j],train_no_baller_means[j],train_no_baller_stdvs[j])), 10)
      pred = 1 if (baller > noballer) else 0
      confmat[int(pred)][int(correct_class)]+=1
      if int(pred) == int(correct_class): score+=1
    confmats.append(confmat/len(testData))
    scores[x]=score/len(testData)   

  #plot the first confusion matrix and calc stats
  cm=confmats[0]
  print(cm)
  print(scores)

  plt.xlabel('Trial')
  plt.ylabel('Percent of Correct Predictions', fontsize=14)
  plt.title('Overall Predictive Percent: Bayes', fontsize=14)
  img =plt.imread("court.jpg")
  plt.imshow(img, zorder=0, extent=[-5,100, 30,100])
  plt.scatter(np.arange(len(scores)), scores*100, c='black', zorder=1, )
  plt.show()
  plot_confmat(cm)

  TP,FP,FN,TN=cm[0][0],cm[0][1],cm[1][0],cm[1][1]
  accuracy =(TP+TN)/(TP+FP+FN+TN)
  precision=TP/(TP+FP)
  recall   =TP/(TP+FN)  
  print(f'accuracy  = {accuracy} ')
  print(f'precision = {precision}')
  print(f'recall    = {recall}   ')
  timestamp(timeStart)

main()
