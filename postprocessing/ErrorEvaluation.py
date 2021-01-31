# Authors: Markus Laubenthal, Lennard Alms

import numpy as np

def calculate_errors(pred_in, base_in, internet_min, internet_max, max_lookback=168, test_size=168, log10 = True):
    pred = pred_in.copy()
    pred = pred.reshape((pred.shape[0],10000)).T
    pred = pred * (internet_max - internet_min) + internet_min
    if log10:
        pred = np.power(np.full(pred.shape, 10), pred) - 1

    base = base_in[:,max_lookback:]

    print('all: ', np.sqrt(((pred-base)**2).mean()))
    print('test: ', np.sqrt(((pred[:,:-test_size]-base[:,:-test_size])**2).mean()))
    print('val: ', np.sqrt(((pred[:,-test_size:]-base[:,-test_size:])**2).mean()))

    return pred, base
