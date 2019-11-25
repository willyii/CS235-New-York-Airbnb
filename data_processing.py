import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import  seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
Address='/Users/zengdekang/Desktop/new-york-city-airbnb-open-data/AB_NYC_2019.csv'

def OneHotEncoding(x):
    #https://blog.csdn.net/m0_37324740/article/details/77169771
    return np.array(pd.get_dummies(x))

def Read_Data(x):
    return pd.read_csv(x)

def Max_min_norm(x):
    return np.array((x.values-x.min())/(x.max()-x.min())).reshape(-1,1)

def Log_norm(x):
    return np.array(np.log(x+1)/np.log(x.max())).reshape(-1,1)

def Fillna_with_Min(x):
    return x.fillna(x.min())

def Fillna_with_Median(x):
    return x.fillna(x.median())

def Data_Processing():
    #read data from csv
    csv=Read_Data(Address)

    #One_Hot_encoding for group, neibourhood and room type
    Group=OneHotEncoding(csv['neighbourhood_group'])
    Neibourhood=OneHotEncoding(csv['neighbourhood'])
    Roomtype=OneHotEncoding(csv['room_type'])

    #max-min norm
    latitude=Max_min_norm(csv['latitude'])
    longitude=Max_min_norm(csv['longitude'])
    host_listings_count=Max_min_norm(csv['calculated_host_listings_count'])
    number_of_reviews=Max_min_norm(csv['number_of_reviews'])
    availability=Max_min_norm(csv['availability_365'])
    nights=Log_norm(csv['minimum_nights'])

    #processing for last_review
    csv['last_review']=pd.to_datetime(csv['last_review'])
    csv['last_review'] = (csv['last_review'].max() - csv['last_review']).dt.days
    csv['last_review']=Fillna_with_Median(csv['last_review'])
    last_view=Log_norm(csv['last_review'])

    #processing for reviews_pre_month
    csv['reviews_per_month']=Fillna_with_Min(csv['reviews_per_month'])
    reviews_per_month=Log_norm(csv['reviews_per_month'])
    price=np.array(csv['price']).reshape(-1,1)
    
    data=np.hstack((Group,Neibourhood,Roomtype,latitude,longitude,nights,number_of_reviews,host_listings_count,availability,last_view,reviews_per_month,price))
    np.save('/Users/zengdekang/Desktop/data.npy',data)

Data_Processing()
