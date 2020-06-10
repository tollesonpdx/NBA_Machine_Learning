import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
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
    fixedNaNs = pd.read_csv(os.path.join(__location__,'./SourceData/fixed_nans.csv'))

    if (v):
        print(f"playerData shape:{playerData.shape}")
        print(f"players shape:{players.shape}")
        print(f"seasonsStats shape:{seasonsStats.shape}")
        print(f'tom\'s stuff shape:{tomsStuff.shape}')
        print(f'fixed NaNs shape:{fixedNaNs.shape}')
    if (v):
        print(playerData)
        print(players)
        print(seasonsStats)
        print(glossary)
        print(tomsStuff)
        print(fixedNaNs)

    return playerData, players, seasonsStats, glossary, nbaNCAABplayers, tomsStuff, fixedNaNs

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
    
    # this is for the NCAA data
    # source = shuffle(source.drop(columns=['name','college','height','birth_date','position','url'])) 
    
    # this is for Tom's combined data and/or the fixed-NaNs data
    source['success'] = [1 if x>=ng else 0 for x in source['nba_gms_plyed']]
    if True:
        source = source.drop(columns=['nba_gms_plyed'])
        # the player, name, and college fields were dropped before loading
    else:
        # source.set_index(['player'], inplace=True)
        source = source.drop(columns=['player', 'name','college','nba_gms_plyed'])
        source = source.fillna(source.mean()) # seems like this may not work
        source = source.fillna(1)

    denomanoms = source.abs().max()
    maxYear = denomanoms['draft_yr']
    source = shuffle(source / denomanoms)
    
    if split > 1: # use this if we are splitting based on year
        trainingData = source[source['draft_yr'] <= split/maxYear]
        trainingTargets = trainingData['success']
        trainingData.drop(columns=['success','draft_yr'])
        
        testingData = source[source['draft_yr'] > split/maxYear]
        testingTargets = testingData['success']
        testingData.drop(columns=['success','draft_yr'])

    else: # use this for a % split between training and testing populations
        lenTraining = int(len(source) * split)

        trainingData = source[:lenTraining]
        trainingTargets = trainingData['success']
        trainingData = trainingData.drop(columns=['success','draft_yr'])
        
        testingData = source[lenTraining:]
        testingTargets = testingData['success']
        testingData = testingData.drop(columns=['success','draft_yr'])

    trainingData = trainingData.values
    trainingTargets = trainingTargets.values
    testingData = testingData.values
    testingTargets = testingTargets.values

    if (v):
        print(f'max values for source table: {denomanoms.shape}')
        print(f'training data and targets: {trainingData.shape, trainingTargets.shape}')
        print(f' testing data and targets: {testingData.shape, testingTargets.shape}')

    return trainingData, trainingTargets, testingData, testingTargets

def plotSVM():
    # # plot the decision function for each datapoint on the grid
    range = 2
    # xx, yy = np.meshgrid(np.linspace(-range, range, 500),
                        # np.linspace(-range, range, 500))

    # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)

    # plt.imshow(Z, interpolation='nearest',
    #         extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
    #         origin='lower', cmap=plt.cm.PuOr_r)
    # contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
    #                     linestyles='dashed')
    # plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
    #             edgecolors='k')
    # plt.xticks(())
    # plt.yticks(())
    # plt.axis([-range, range, -range, range])
    # plt.show()

def printConfusionMatrix(matrixIn):
    """prints the confusion matrix to standard output"""
    count = 0
    print(f"\nConfusion Matrix:\n    ",end='')
    for i in range(len(matrixIn)):
        print(str(i).rjust(7,' '),end='')
    print()
    for row in matrixIn:
        print(count,end='   ')
        for item in row:
            print(str(item).rjust(7,' '),end='')
        print()
        count += 1

def SVM(x_train, y_train, x_test, y_test):

    lenTest = len(y_test)
    clf = svm.NuSVC(nu=.6, gamma='auto')
    # clf = svm.SVC(kernel='rbf', gamma='auto')
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    confusion=np.zeros((2,2))
    for i in range(lenTest):
        confusion[int(pred[i])][int(y_test[i])]+=1

    accuracy = round(accuracy_score(y_test, pred) * 100, 2)

    return accuracy, confusion


if __name__ == "__main__":
    timeStart = time.time()

    ### hyperparameters ###
    v = 0 # flag for verbose printing
    ng = 175 # number of games played needed to be a "successful" player
    split = 0.8 # proportion of training data to whole
    year = 2009 # year cutoff if using years for training vs testing
    rounds = 10

    playerData, players, seasonsStats, glossary, nbaNCAABplayers, tomsStuff, fixedNaNs = loadData()
    
    results = []
    for i in range(1,rounds+1):
        x_train, y_train, x_test, y_test = prepAndSplitData(fixedNaNs, split, ng)
        accuracy, confMatrix = SVM(x_train, y_train, x_test, y_test)
        # print(f'Percent correctly predicted by SVM model: {correctPct}%')
        print(f'{accuracy}%', end=' ')
        if i%10 == 0: print()
        results.append([i, accuracy])
        # printConfusionMatrix(confMatrix)

    # print(results)

    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - timeStart), 60)
    print(f'This program took {int(minutes)} minutes and {(seconds)} seconds to run.')