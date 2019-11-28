import pandas as pd
import numpy as np


class Log_Norm:
    def __init__(self):
        self.Max=0


    def fit(self, data: pd.DataFrame,column_name: str):

        self.Max = data[column_name].max()
   

    def transform(self,data: pd.DataFrame,column_name: str):
        #change new maximum value to original maximum value
        data.loc[data[column_name]>self.Max] =self.Max

        return np.log(data[column_name])/np.log(self.Max)


if __name__ == '__main__':
    trainset= pd.DataFrame({'A': [2,3,4,4,5]})
    testset = pd.DataFrame({'A': [6,1]})
    test = Log_Norm()
    test.fit(trainset, 'A')
    a=test.transform(trainset,'A')
    b=test.transform(testset, 'A')
    print(a)
    print(b)