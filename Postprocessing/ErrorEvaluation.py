def calculate_errors(pred_in, base_in, max_lookback=168, test_size=168):
    pred = pred_in.copy()
    pred = pred.reshape((pred.shape[0],10000)).T
    pred = pred * (internet_max - internet_min)[:,np.newaxis] + internet_min[:,np.newaxis]
    pred = np.power(np.full(pred.shape, 10), pred) - 1

    base = base_in[:,max_lookback:]

    print('all: ', np.sqrt(((pred-base)**2).mean()))
    print('test: ', np.sqrt(((pred[:,:-test_size]-base[:,:-test_size])**2).mean()))
    print('val: ', np.sqrt(((pred[:,-test_size:]-base[:,-test_size:])**2).mean()))

    return pred, base
