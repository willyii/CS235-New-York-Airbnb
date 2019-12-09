from KNN.KNN_cs235 import KNN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # Generate the random dataset. The label equals to the sum of every feature plus the noise
    dataset = np.random.rand(5000, 60)
    label = np.sum(dataset, axis=1).reshape((5000, 1))+ np.random.rand(5000,1)
    dataset_df = pd.DataFrame(dataset)

    X_train, X_test, Y_train, Y_test = train_test_split(dataset_df, label, test_size=0.1)


    knn = KNN()

    knn.fit(X_train.to_numpy(), Y_train)
    predicted = knn.predict(X_test.to_numpy())
    print("MSE:", knn.Loss(predicted, Y_test))
