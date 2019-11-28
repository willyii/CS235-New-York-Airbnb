import pandas as pd
import numpy as np


class Z_Score_Norm:
    def __init__(self):
        self.Mean=0
        self.Std=0
        self.Max=0
        self.Min=0

    def fit(self, data: pd.DataFrame,column_name: str):
        self.Mean=data[column_name].mean()
        self.Std=data[column_name].std()
        self.Max = data[column_name].max()
        self.Min = data[column_name].min()

    def transform(self,data: pd.DataFrame,column_name: str):
        #change new maximum value to original maximum value
        data.loc[data[column_name]>self.Max] =self.Max
        # change new minimum value to original minimum value
        data.loc[data[column_name] < self.Min] = self.Min

        return np.array((data[column_name].values-self.Mean)/self.Std)


if __name__ == '__main__':
    trainset= pd.DataFrame({'A': [1,3,4,4,5]})
    testset = pd.DataFrame({'A': [6,0]})
    test = Z_Score_Norm()
    test.fit(trainset, 'A')
    a=test.transform(trainset,'A')
    b=test.transform(testset, 'A')
    print(a)
    print(b)