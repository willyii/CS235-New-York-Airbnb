import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import  seaborn as sns

Address='../Data/AB_NYC_2019.csv'
class Data_processing(object):
    def __init__(self,data_url):
        #load csv file
        self.data=pd.read_csv(data_url)

    def OneHotEncoding(self,column_name):
        #OneHotEncoding for one column and return Numpy
        # https://blog.csdn.net/m0_37324740/article/details/77169771
        return np.array(pd.get_dummies(self.data[column_name]))

    def Max_min_norm(self,column_name):
        # X-X.min/X.max-X.min
        # return one dimension Numpy
        return np.array((self.data[column_name].values-self.data[column_name].min())/(self.data[column_name].max()-self.data[column_name].min())).reshape(-1,1)

    def Log_norm(self,column_name):
        #Log(x+1)/log(x.max) to 0~1
        #return one dimension Numpy
        return np.array(np.log(self.data[column_name]+1)/np.log(self.data[column_name].max())).reshape(-1,1)

    def Log_1pNorm(self,column_name):
        # Log(x+1)
        # return one dimension Numpy
        return np.array(np.log1p(self.data[column_name])).reshape(-1,1)

    def Z_Score(self,column_name):
        # x-x.mean()/x.std()
        # return one dimension Numpy
        return np.array((self.data[column_name].values-self.data[column_name].mean())/self.data[column_name].std()).reshape(-1,1)

    def Fillna_with_Min(self,column_name):
        #fill with minimum value
        return self.data[column_name].fillna(self.data[column_name].min())

    def Fillna_with_Zero(self,column_name):
        # fill with 0
        return self.data[column_name].fillna(0)

    def Fillna_with_Median(self,column_name):
        return self.data[column_name].fillna(self.data[column_name].median())

    def Fillna_with_Max(self,column_name):
        return self.data[column_name].fillna(self.data[column_name].max())

    def To_Numpy(self,column_name):
        #change a column in csv to Numpy
        return np.array(self.data[column_name]).reshape(-1,1)



