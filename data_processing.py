import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def OneHotEncoding(array,x):

    group=np.unique(array[:,x])
    count=group.shape[0]
    print(count)
    transformed_data=[]
    transpose_data=np.transpose(array)
    one_hot_matrix=np.eye(count)
    for i in tqdm(array[:,x]):
        for j in range(count):
            if i==group[j]:transformed_data.append(one_hot_matrix[j])
    print(one_hot_matrix)
    transformed_data=np.array(transformed_data)
    for i in range(transformed_data.shape[1]):
        transpose_data=np.insert(transpose_data,x+i+1,transformed_data[:,i],0)
        print(transpose_data)
    transpose_data=np.delete(transpose_data,x,0)
    return np.transpose(transpose_data)


#read date from csv

f=pd.read_csv('/Users/zengdekang/Desktop/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#https://blog.csdn.net/m0_37324740/article/details/77169771
#One_Hot_encoding for group, neibourhood and room type
group_ohe=np.array(pd.get_dummies(f['neighbourhood_group']))
print(group_ohe)
neibour_ohe=np.array(pd.get_dummies(f['neighbourhood']))
roomtype_ohe=np.array(pd.get_dummies(f['room_type']))

#max-min norm
latitude=( f['latitude'].values - f['latitude'].min() ) / ( f['latitude'].max() - f['latitude'].min() )
latitude=np.array(latitude).reshape(-1,1)

longtitude=( f['longitude'].values - f['longitude'].min() ) / ( f['longitude'].max() - f['longitude'].min() )
longtitude=np.array(longtitude).reshape(-1,1)

host_listings_count=( f['calculated_host_listings_count'].values - f['calculated_host_listings_count'].min() )\
                               / f['calculated_host_listings_count'].max() - f['calculated_host_listings_count'].min()
host_listings_count=np.array(host_listings_count).reshape(-1,1)

number_of_reviews=( f['number_of_reviews'].values - f['number_of_reviews'].min() ) /\
                  ( f['number_of_reviews'].max() - f['number_of_reviews'].min() )
number_of_reviews=np.array(number_of_reviews).reshape(-1,1)

availability=( f['availability_365'].values - f['availability_365'].min() ) /\
                  ( f['availability_365'].max() - f['availability_365'].min() )
availability=np.array(availability).reshape(-1,1)
#z-score norm

price=(f['price'].values-f['price'].mean())/f['price'].std()
nights=(f['minimum_nights'].values-f['minimum_nights'].mean())/f['minimum_nights'].std()
price=np.array(price).reshape(-1,1)
nights=np.array(nights).reshape(-1,1)
#max-min norm
f['last_review']=pd.to_datetime(f['last_review'])
f['last_review']=f['last_review'].fillna(f['last_review'].min())
last_view=( f['last_review'] - f['last_review'].min() ) /\
                  ( f['last_review'].max() - f['last_review'].min() )
print(f['last_review'].min())
last_view=np.array(last_view).reshape(-1,1)
#max-min norm
f['reviews_per_month']=f['reviews_per_month'].fillna(f['reviews_per_month'].min())
reviews_per_month=( f['reviews_per_month'] - f['reviews_per_month'].min() ) /\
                  ( f['reviews_per_month'].max() - f['reviews_per_month'].min() )
reviews_per_month=np.array(reviews_per_month).reshape(-1,1)

data=np.hstack((group_ohe,neibour_ohe,latitude,longtitude,roomtype_ohe,nights,number_of_reviews,last_view,reviews_per_month
                 ,host_listings_count,availability,price))





#csvrows=csv.reader(f)
#for row in csvrows:
#    i.append(row)
#convert to array
#data=np.array(i)

#delete columns: id, name, host_name, host_id
#data=data[1:,4:]
#print(data)
#a=OneHotEncoding(data,1)
#f=open('/Users/zengdekang/Desktop/new-york-city-airbnb-open-data/AB_NYC_2019_1.csv','w')
##   f.writelines(row)


#b=OneHotEncoding(a,5)
#print(a)
