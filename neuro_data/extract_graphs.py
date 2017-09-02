import sys
import time
import logging
import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration as FA
from sklearn.decomposition import PCA
import pandas as pd
import networkx as nx

import icstar
from data import extract_neurodata

MAX_DIM = 100

if __name__ == "__main__":
    logging.basicConfig(
        filename="{}.log".format(time.time()), level=logging.INFO)

    # Load the data.
    data = extract_neurodata.make_ralph_data()

    # Reduce dimensionality by random sampling of features.
    for key in data.keys():
        print('Preparing {}...'.format(key))
        voxels = data[key]
        if voxels.shape[1] > MAX_DIM:
            data[key] = PCA(n_components=MAX_DIM).fit_transform(voxels)
    variable_types = dict([(vname, 'c') for vname in data.keys()])

    # Initialize IC*.
    testargs = {'verbose': True,
            'max_dim': 100,
            'logdim': False,
            'num_perm': 8}
    ic = icstar.IC(alpha=0.01, testargs=testargs)

    # Run the search algorithm.
    graph = ic.search(data, variable_types)
    print(graph.edges())
    pkl.dump(graph, open('graph100.pkl', 'w'), protocol=pkl.HIGHEST_PROTOCOL)
    
    # Draw the graph.
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos=pos, with_labels=True)
    plt.savefig('graph100.png')
