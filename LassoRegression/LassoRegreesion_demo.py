import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from LassoRegression.LassoRegression_CS235 import Lasso
from sklearn.preprocessing import Normalizer, MinMaxScaler

if __name__ == '__main__':
    dataset = np.random.rand(5000, 60)
    label = np.sum(dataset, axis=1).reshape((5000, 1))+ np.random.rand(5000,1)
    dataset_df = pd.DataFrame(dataset)


    X_train, X_test, Y_train, Y_test = train_test_split(dataset, label, test_size=0.1)


    max_min = MinMaxScaler()
    max_min.fit(Y_train)
    Y_train = max_min.transform(Y_train)
    Y_test = max_min.transform(Y_test)


    nm = Normalizer()
    nm.fit(X_train)
    X_train = nm.transform(X_train)
    X_test = nm.transform(X_test)


    lasso = Lasso()
    lasso.fit(X_train, Y_train)
    x = lasso.predict(X_test)
    print(x)
    # print("MES:",lasso.error(X_test, Y_test))