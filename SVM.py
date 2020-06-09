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

def prepAndSplitData(source, split, ng):
    lenTraining = int(len(source) * split)
    
    # this is for the NCAA data
    # source = shuffle(source.drop(columns=['name','college','height','birth_date','position','url'])) 
    
    # this is for Tom's combined data
    source.set_index(['player'], inplace=True)
    source['success'] = [1 if x>=ng else 0 for x in source['nba_gms_plyed']]
    source = shuffle(source.drop(columns=['name','college','nba_gms_plyed']))

    denomanoms = source.abs().max()
    maxYear = denomanoms['draft_yr']
    source = source/denomanoms
    
    if (v): print(denomanoms)
    
    trainingData = source[source['draft_yr'] <= split/maxYear]
    testingData = source[source['draft_yr'] > split/maxYear]
    return trainingData, testingData

if __name__ == "__main__":
    timeStart = time.time()

    ### hyperparameters ###
    v = 0 # flag for verbose printing
    ng = 240 # number of games played needed to be a "successful" player
    split = 2009 # proportion of training data to whole
    

    playerData, players, seasonsStats, glossary, nbaNCAABplayers, tomsStuff = loadData()
    successfulPlayers = idSuccessNew(seasonsStats, ng)
    trainData, testData = prepAndSplitData(tomsStuff, split, ng)
    print(trainData)
    # SVM goes here
    if False:
        # print(__doc__)

        range = 2
        xx, yy = np.meshgrid(np.linspace(-range, range, 500),
                            np.linspace(-range, range, 500))
        np.random.seed(0)
        X = np.random.randn(300, 2)
        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

        # fit the model
        clf = svm.NuSVC(gamma='auto')
        clf.fit(X, Y)

        # plot the decision function for each datapoint on the grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.imshow(Z, interpolation='nearest',
                extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                origin='lower', cmap=plt.cm.PuOr_r)
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                            linestyles='dashed')
        plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis([-range, range, -range, range])
        plt.show()

    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - timeStart), 60)
    print(f'This program took {int(minutes)} minutes and {(seconds)} seconds to run.')