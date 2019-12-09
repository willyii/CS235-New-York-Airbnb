from RrandomForests.Random_Forests_CS235 import RandomForests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    dataset = np.random.rand(5000, 60)
    label = np.sum(dataset, axis=1).reshape((5000, 1))+ np.random.rand(5000,1)
    dataset_df = pd.DataFrame(dataset)


    X_train, X_test, Y_train, Y_test = train_test_split(dataset_df, label, test_size=0.1)

    train_data = np.hstack((X_train, Y_train))
    test_data = np.hstack((X_test, Y_test))

    n_folds = 5
    max_depth = 5
    min_size = 1
    sample_size = 0.7
    n_trees = 5
    n_features = dataset.shape[1]
    rf = RandomForests(n_folds, max_depth, min_size, sample_size, n_trees, dataset)

    scores = rf.accuracy_metric(Y_test,rf.random_forest(train_data, test_data))
    print('Mean Squared error:' , scores)