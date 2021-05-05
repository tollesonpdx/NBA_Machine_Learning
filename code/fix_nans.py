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

def fix_missings(source):

  lenTraining=int(len(source))  
  source=shuffle(source.drop(columns=['name','college','success','player']))  
  no_nans=source.dropna(axis=0, subset=['hand_length','wingspan'])
  no_nans=no_nans.loc[:,['hand_length','wingspan']]
  missing_handlen=source['hand_length'].isnull()
  wing_miss_handlen=pd.DataFrame(source['wingspan'][missing_handlen])
  X=no_nans[['wingspan']]
  y=no_nans['hand_length']
  X_train=X[:lenTraining]
  y_train=y[:lenTraining]
  lm=LinearRegression().fit(X_train, y_train)
  hand_len_pred=lm.predict(wing_miss_handlen)
  source.loc[source['hand_length'].isnull(), 'hand_length']=hand_len_pred
  
  no_nans=source.dropna(axis=0, subset=['vert_maxreach','wingspan'])
  no_nans=no_nans.loc[:,['vert_maxreach','wingspan']]
  missing_vmr=source['vert_maxreach'].isnull()
  wing_miss_vmr=pd.DataFrame(source['wingspan'][missing_vmr])
  X=no_nans[['wingspan']]
  y=no_nans['vert_maxreach']
  X_train=X[:lenTraining]
  y_train=y[:lenTraining]
  lm=LinearRegression().fit(X_train, y_train)
  vmr_pred=lm.predict(wing_miss_vmr)
  source.loc[source['vert_maxreach'].isnull(), 'vert_maxreach']=vmr_pred

  no_nans=source.dropna(axis=0, subset=['vert_max','standing_reach'])
  no_nans=no_nans.loc[:,['vert_max','standing_reach']]
  missing_vm=source['vert_max'].isnull()
  stand_miss_vm=pd.DataFrame(source['standing_reach'][missing_vm])
  X=no_nans[['standing_reach']]
  y=no_nans['vert_max']
  X_train=X[:lenTraining]
  y_train=y[:lenTraining]
  lm=LinearRegression().fit(X_train, y_train)
  vm_pred=lm.predict(stand_miss_vm)
  source.loc[source['vert_max'].isnull(), 'vert_max']=vm_pred

  no_nans=source.dropna(axis=0, subset=['vert_nostep','standing_reach'])
  no_nans=no_nans.loc[:,['vert_nostep','standing_reach']]
  missing_vns=source['vert_nostep'].isnull()
  stand_miss_vns=pd.DataFrame(source['standing_reach'][missing_vns])
  X=no_nans[['standing_reach']]
  y=no_nans['vert_nostep']
  X_train=X[:lenTraining]
  y_train=y[:lenTraining]
  lm=LinearRegression().fit(X_train, y_train)
  vns_pred=lm.predict(stand_miss_vns)
  source.loc[source['vert_nostep'].isnull(), 'vert_nostep']=vns_pred

  no_nans=source.dropna(axis=0, subset=['hand_width','hand_length'])
  no_nans=no_nans.loc[:,['hand_width','hand_length']]
  missing_hand_wid=source['hand_width'].isnull()
  stand_miss_hand_wid=pd.DataFrame(source['hand_length'][missing_hand_wid])
  X=no_nans[['hand_length']]
  y=no_nans['hand_width']
  X_train=X[:lenTraining]
  y_train=y[:lenTraining]
  lm=LinearRegression().fit(X_train, y_train)
  hand_wid_pred=lm.predict(stand_miss_hand_wid)
  source.loc[source['hand_width'].isnull(), 'hand_width']=hand_wid_pred
  
  source.to_csv('fixed_nans.csv',index=False)

def main():
  timeStart = time.time()
  toms_data=pd.read_csv(os.path.join(__location__,'./SourceData/toms_combined_data.csv'))
  toms_data=fix_missings(toms_data) 
  timestamp(timeStart)

main()
