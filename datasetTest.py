import os
import time
import pandas as pd

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def loadData():
    """load the source data"""
    seasonsStats = pd.read_csv(os.path.join(__location__,'./SourceData/Seasons_Stats.csv'), index_col=0)
    nbaNCAABplayers = pd.read_csv(os.path.join(__location__,'./SourceData/nba_ncaab_players.csv'))
    combineData = pd.read_csv(os.path.join(__location__,'./SourceData/nba_draft_combine_all_years.csv'))
    return seasonsStats, nbaNCAABplayers, combineData

if __name__ == "__main__":

    seasonsStats, nbaNCAABplayers, combineData = loadData()

<<<<<<< HEAD
    if True:
        seasStat_Group = (seasonsStats.groupby(['Player', 'Tm'])['Year'].count())
        seasStat_Group.sort_values(ascending=False, inplace=True)
        print(f'season stats shape:{seasStat_Group.shape}')
        print(seasStat_Group.head())
        print(seasonsStats.loc[seasonsStats['Player']=='Kobe Bryant'])

    if False:
        ncaab_Group = nbaNCAABplayers.groupby(['name'])['NBA_g_played', 'NCAA_games'].count()
        ncaab_Group.sort_values(by='NBA_g_played', ascending=False, inplace=True)
        print(f'ncaab shape:{ncaab_Group.shape}')
        print(ncaab_Group.head())
=======
    activeFrom = [2010,2011,2012,2013,2014,2015]
    combineYears = [x-1 for x in activeFrom]
    
    seasStat_Group = (seasonsStats.groupby(['Player'])['Year'].count())
    print(f'season stats shape:{seasStat_Group.shape}')

    ncaab_Group = nbaNCAABplayers.groupby(['pid','name'])['NBA_g_played'].count()
    print(f'ncaab shape:{ncaab_Group.shape}')

    nbaNCAABplayers = nbaNCAABplayers[nbaNCAABplayers.active_from.isin(activeFrom)]
    combineData = combineData[combineData.Year.isin(combineYears)]
    print(f'combine data shape:{combineData.shape}')
>>>>>>> origin/master
