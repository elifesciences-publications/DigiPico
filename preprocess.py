from __future__ import print_function
import numpy as np
import pandas


def prep_data(path = 'Data/Filter.csv'):

    dataframe = pandas.read_csv(path, header=None)
    dataset = dataframe.values

    X = dataset[:, 2:].astype(float)
    Y = dataset[:, 1].astype(int)

    X = np.nan_to_num(X)





