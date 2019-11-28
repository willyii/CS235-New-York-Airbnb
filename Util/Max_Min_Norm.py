import pandas as pd
import numpy as np


class Max_Min_Norm:
    def __init__(self):
        self.Max=0
        self.Min=0

    def fit(self, data: pd.DataFrame,column_name: str):
        self.Max=data[column_name].max()
        self.Min=data[column_name].min()

    def transform(self,data: pd.DataFrame,column_name: str):

        data.loc[data[column_name]>self.Max] =self.Max
        data.loc[data[column_name] < self.Min] = self.Min

        return np.array((data[column_name].values-self.Min)/
                        (self.Max-self.Min))

if __name__ == '__main__':
    trainset= pd.DataFrame({'A': [1,2,3,4,5]})
    testset = pd.DataFrame({'A': [6,0]})
    test = Max_Min_Norm()
    test.fit(trainset, 'A')
    test.transform(trainset,'A')
    test.transform(testset, 'A')

    print(testset)