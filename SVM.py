import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import shuffle

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def loadData():
    """load the source data"""
    playerData = pd.read_csv(os.path.join(__location__,'./SourceData/player_data.csv'))
    players = pd.read_csv(os.path.join(__location__,'./SourceData/Players.csv'))
    seasonsStats = pd.read_csv(os.path.join(__location__,'./SourceData/Seasons_Stats.csv'), index_col=0)
    glossary = dict(pd.read_csv(os.path.join(__location__,'./SourceData/Seasons_Stats_Glossary.txt'),
                           sep='|', names=['abbv','description'], skip_blank_lines=True).values)
    nbaNCAABplayers = pd.read_csv(os.path.join(__location__,'./SourceData/nba_ncaab_players.csv'))
    tomsStuff = pd.read_csv(os.path.join(__location__,'./SourceData/toms_combined_data.csv'))

    if (v):
        print(f"playerData shape:{playerData.shape}")
        print(f"players shape:{players.shape}")
        print(f"seasonsStats shape:{seasonsStats.shape}")
    if (v):
        print(playerData)
        print(players)
        print(seasonsStats)
        print(glossary)
        # seasonsStats.rename(columns = glossary, inplace=True)

    return playerData, players, seasonsStats, glossary, nbaNCAABplayers, tomsStuff

def idSuccessOld(seasonsStats, ng):
    playerGames={}
    success_players=[]
    if (v): print(seasonsStats['G'].head())
    for player in seasonsStats['Player']:
      if player not in playerGames.keys():
        playerGames[player]=0
        games=[seasonsStats.loc[seasonsStats['Player']==player, 'G']][0].values
        playerGames[player]=sum(games)
        if playerGames[player] >= ng:
          if (v): print(player, playerGames[player])
          success_players.append(player)

def idSuccessNew(seasonsStats, ng):
    successP = seasonsStats.groupby(['Player'])['G'].sum()
    successP.columns = ['Player', 'G']
    successP = successP >= ng
    if (v): print(successP)
    return successP

def prepAndSplitData(source, split):
    lenTraining = int(len(source) * split)
    
    # this is for the NCAA data
    # source = shuffle(source.drop(columns=['name','college','height','birth_date','position','url'])) 
    
    # this is for Tom's combined data
    source = shuffle(source.drop(columns=['name','college'])) 
    
    denomanoms = source.abs().max()
    source = source/denomanoms
    print(source)
    trainingData = source[:lenTraining]
    testingData = source[lenTraining:]
    return trainingData, testingData

if __name__ == "__main__":
    timeStart = time.time()

    ### hyperparameters ###
    v = 0 # flag for verbose printing
    ng = 174 # number of games played needed to be a "successful" player
    split = .8 # proportion of training data to whole
    

    playerData, players, seasonsStats, glossary, nbaNCAABplayers, tomsStuff = loadData()
    successfulPlayers = idSuccessNew(seasonsStats, ng)
    trainData, testData = prepAndSplitData(tomsStuff, split)

    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - timeStart), 60)
    print(f'This program took {int(minutes)} minutes and {(seconds)} seconds to run.')