


def seven_days_train_test_split(dataset, max_lookback):
  train_indices = range(0,X_closeness.shape[0] - max_lookback)
  test_indices = range(X_closeness.shape[0] - max_lookback, X_closeness.shape[0])
  dataset_train, dataset_test = dataset[train_indices], dataset[test_indices]
  return dataset_train, dataset_test
