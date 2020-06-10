import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

  # scaler = StandardScaler()
  # scaler.fit(X_train) 
  # X_train=pd.DataFrame(scaler.transform(X_train),columns = X_train.columns)  
  
  lm = LogisticRegression(random_state = 0)
  lm.fit(X_train, y_train)
  # lm=LogisticRegression().fit(X_train, y_train)
  pred=lm.predict(X_test)
  correct=0
  for i in range(len(pred)):
    if y_test[i]==pred[i]:
      correct+=1
  print(f'Percent correctly predicted by logistic regression model: {round(correct/len(y_test), 2)}%')
  # print('Coefficients: \n', lm.coef_)
  # print(f'Mean squared error: {round(mean_squared_error(y_test, pred), 2)}')
  # # The coefficient of determination: 1 is perfect prediction
  # print('Coefficient of determination: %.2f'
  #       % r2_score(y_test, pred))


  feature_importance=abs(lm.coef_[0])
  feature_importance=100.0*(feature_importance/feature_importance.max())
  sorted_idx = np.argsort(feature_importance)
  pos=np.arange(sorted_idx.shape[0]) + .5

  featfig=plt.figure()
  featax=featfig.add_subplot(1, 1, 1)
  featax.barh(pos, feature_importance[sorted_idx], align='center',color=plot_colors,edgecolor='black')
  featax.set_yticks(pos)
  featax.set_yticklabels(np.array(X_train.columns)[sorted_idx], fontsize=12)
  featax.set_xlabel('Relative Feature Importance For NBA Success', fontsize=14)

  plt.tight_layout()   
  plt.show()
  # Plot outputs
  # print(X_test.shape, y_test.shape)
  # print(X_test)
  # plt.scatter(X_test, y_test,  color='black')
  # plt.plot(X_test, pred, color='blue', linewidth=3)

  # plt.xticks(())
  # plt.yticks(())

  # plt.show()

def main():
  timeStart = time.time()
  split=.8
  toms_data = pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))
  trainData, testData = prep_data(toms_data, split)
  regression(trainData, testData)
  
  timestamp(timeStart)

main()
