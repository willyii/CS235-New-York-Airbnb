import numpy as np

dataset = load_data(path)
n_folds = 5
max_depth = 5
min_size = 1
sample_size = 0.7
n_trees = 5
method = RandomForests(n_folds, max_depth, min_size, sample_size, n_trees, dataset.to_numpy())
scores = method.random_forest()
print('Mean Squared error: %.3f%%' % np.mean(scores))