import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LassoRegression.LassoRegression_CS235 import Lasso
from sklearn.preprocessing import Normalizer, MinMaxScaler


def normalize(matrix):
    norm = (np.linalg.norm(matrix, axis=0))
    norm[norm == 0] = 1
    normalized_matrix = matrix / norm
    return normalized_matrix, norm


if __name__ == '__main__':
    dataset = np.random.rand(5000, 60)
    label = np.sum(dataset, axis=1).reshape((5000, 1))+ np.random.rand(5000,1)
    dataset_df = pd.DataFrame(dataset)


    X_train, X_test, Y_train, Y_test = train_test_split(dataset, label, test_size=0.1)


    # Normalize
    nm = Normalizer()
    nm.fit(X_train)
    X_train, norm = normalize(X_train)
    X_test = X_test/norm


    lasso = Lasso()
    lasso.fit(X_train, Y_train)
    x = lasso.predict(X_train)
    print("MES:",lasso.error(X_test, Y_test))