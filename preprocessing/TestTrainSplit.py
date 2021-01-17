


def seven_days_train_test_split(dataset, max_lookback):
  train_indices = range(0,dataset.shape[0] - max_lookback)
  test_indices = range(dataset.shape[0] - max_lookback, dataset.shape[0])
  dataset_train, dataset_test = dataset[train_indices], dataset[test_indices]
  return dataset_train, dataset_test
