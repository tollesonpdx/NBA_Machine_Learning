import os
import time
import pandas as pd
import numpy as np


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def loadData():
    """load the source data"""
    
    # load source data from the source files
    gaussians = pd.read_csv(os.path.join(__location__,'./cluster_dataset.txt'), 
                            engine='python', sep="  ", header=None,
                            names=["x","y"])
    
    if (v):
        print(f"gaussians shape:{gaussians.shape}\nmax:\n{gaussians.max()}\nmin:\n{gaussians.min()}")
        print(gaussians)
        emptyDF = pd.DataFrame()
        plotStuff(gaussians, 'freshly_loaded_data', 'label', emptyDF, None, None) # plot the loaded data

    return gaussians



if __name__ == "__main__":
    timeStart = time.time()
    v = 1 # flag for verbose printing


    timeEnd = time.time()
    minutes, seconds = divmod((timeEnd - timeStart), 60)
    print(f'This program took {int(minutes)} minutes and {int(seconds)} seconds to run.')