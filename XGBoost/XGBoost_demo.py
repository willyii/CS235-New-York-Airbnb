from XGBoost.XGBoost_CS235 import XGBoost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    dataset = np.random.rand(5000, 60)
    label = np.sum(dataset, axis=1).reshape((5000, 1))+ np.random.rand(5000,1)
    dataset_df = pd.DataFrame(dataset)


    X_train, X_test, Y_train, Y_test = train_test_split(dataset_df, label, test_size=0.1)


    paras = {}
    paras["learning_rate"] = 0.3
    paras["max_depth"] = 4
    paras["subsample"]  = 0.6
    paras["colsample"] = 0.5
    paras["early_stop_round"] = 5
    paras["n_estimator"] = 20
    paras["min_child_weight"] = 10

    xgb_cs235 = XGBoost(paras)

    xgb_cs235.fit(X_train.to_numpy(), Y_train, X_test.to_numpy(), Y_test)
