import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import itertools
import random
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor

def main():

    f = open("fixed_nans.csv", "r")
    parameters = pd.read_csv(f)
    parameters = parameters.iloc[:,:-1]
    parameters = pd.get_dummies(parameters)
    parameters.loc[parameters['nba_gms_plyed']>=240, 'success']=1
    parameters.loc[parameters['nba_gms_plyed'] <240, 'success']=0
    parameters = shuffle(parameters.drop(columns=['nba_gms_plyed']))
    labels = np.array(parameters['success'].values)
    parameters = parameters.drop('success', axis = 1)
    parameters_list = list(parameters.columns)
    train_data, test_data, train_labels, test_labels = train_test_split(parameters, labels, test_size = 0.2, random_state = 87)
    baseline_preds = test_data['draft_pick'].values
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 87)
    rf.fit(train_data, train_labels)
    print(parameters.index)
    predictions = rf.predict(test_data)
    #print(test_labels)
    print(predictions)
    errors = abs(predictions - test_labels)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = 100 * (errors / (1 + predictions))
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Import tools needed for visualization
    from sklearn.tree import export_graphviz
    import pydot
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = parameters_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png('tree.png')

    
   
main()