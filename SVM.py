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
        print(playerData)
        print(players)
        print(seasonsStats)

    return playerData, players, seasonsStats



if __name__ == "__main__":
    timeStart = time.time()
    v = 1 # flag for verbose printing

    playerData, players, seasonsStats = loadData()

    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - timeStart), 60)
    print(f'This program took {int(minutes)} minutes and {int(seconds)} seconds to run.')