import numpy as np
import h5py

import os

def getFileHandler(filename):
    if os.path.isfile(filename):
        f = h5py.File(filename, 'r+')
    else:
        f = h5py.File(filename, 'w')
    return f

def generate_label(grid, max_lookback, f):
    dataset_name = 'label'
    y = grid[max_lookback:]
    y = y.reshape((y.shape[0], -1))

    if dataset_name in f:
        del f[dataset_name]

    h5_dataset = f.create_dataset(dataset_name, y.shape, dtype='f')
    h5_dataset[:] = y[:]
    return h5_dataset


def generate_dataset(grid, lookback, max_lookback, f, dataset_name):
    if dataset_name in f:
        del f[dataset_name]
    h5_dataset = f.create_dataset(dataset_name,
        (grid.shape[0] - max_lookback, grid.shape[1], grid.shape[2], len(lookback)),
        dtype='f',
        chunks=(1, grid.shape[1], grid.shape[2], len(lookback)))
    dataset = np.empty((grid.shape[0] - max_lookback, grid.shape[1], grid.shape[2], len(lookback)))

    for index,i in enumerate(lookback):
        dataset[:,:,:,index] = grid[max_lookback - i:grid.shape[0] - i]
    h5_dataset[:] = dataset[:]
    dataset = None

    return h5_dataset

def get_datasets_from_file(f, dataset_names):
    datasets = []
    for name in dataset_names:
        print(name)
        datasets.append(f[name])
    return datasets
