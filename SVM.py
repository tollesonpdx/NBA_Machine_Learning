import os
import time
import pandas as pd
import numpy as np


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def loadData():
    """load the source data"""
    
    # load source data from the source files
    playerData = pd.read_csv(os.path.join(__location__,'./SourceData/player_data.csv'))
    players = pd.read_csv(os.path.join(__location__,'./SourceData/Players.csv'))
    seasonsStats = pd.read_csv(os.path.join(__location__,'./SourceData/Seasons_Stats.csv'))

    if (v):
        print(f"playerData shape:{playerData.shape}")
        print(f"players shape:{players.shape}")
        print(f"seasonsStats shape:{seasonsStats.shape}")
        # print(playerData.head())
        # print(players.head())
        # print(seasonsStats.head())

    return playerData, players, seasonsStats



if __name__ == "__main__":
    timeStart = time.time()
    v = 1 # flag for verbose printing

    playerData, players, seasonsStats = loadData()

    playerGames={}
    success_players=[]
    print(seasonsStats['G'].head())
    for player in seasonsStats['Player']:
      if player not in playerGames.keys():
        playerGames[player]=0
        games=[seasonsStats.loc[seasonsStats['Player']==player, 'G']][0].values
        playerGames[player]=sum(games)
        if playerGames[player]>=174:
          print(player, playerGames[player])
          success_players.append(player)
    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - timeStart), 60)
    print(f'This program took {int(minutes)} minutes and {int(seconds)} seconds to run.')