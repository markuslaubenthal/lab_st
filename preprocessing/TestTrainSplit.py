def seven_days_train_test_split(dataset, test_size=168):
  train_indices = range(0,dataset.shape[0] - test_size)
  test_indices = range(dataset.shape[0] - test_size, dataset.shape[0])
  dataset_train, dataset_test = dataset[train_indices], dataset[test_indices]
  return dataset_train, dataset_test
