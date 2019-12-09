#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
        
def Load_Data(path):
    global train, price, test, test_pr, norm, prnorm, feature_space, samples
    def OneHotEncoding(x):
        return np.array(pd.get_dummies(x))
    def Max_min_norm(x):
        return np.array((x.values-x.min())/(x.max()-x.min())).reshape(-1,1)
    def Log_norm(x):
        return np.array(np.log(x+1)/np.log(x.max())).reshape(-1,1)
    def Z_Score(x):
        return np.array((x.values-x.mean())/x.std()).reshape(-1,1)
    def Fillna_with_Min(x):
        return x.fillna(x.min())
    def normalize(matrix):
        norm=(np.linalg.norm(matrix, axis = 0))
        norm[norm==0]=1
        normalized_matrix=matrix/norm
        return normalized_matrix, norm
    
    csv = pd.read_csv(path)
    
    ''' 
    Process data using Dekang's original code. Nieghbourhood excluded for better results
    '''
    Group=OneHotEncoding(csv['neighbourhood_group'])
    Roomtype=OneHotEncoding(csv['room_type'])
    latitude=Max_min_norm(csv['latitude'])
    longitude=Max_min_norm(csv['longitude'])
    host_listings_count=Max_min_norm(csv['calculated_host_listings_count'])
    number_of_reviews=Max_min_norm(csv['number_of_reviews'])
    availability=Max_min_norm(csv['availability_365'])
    nights=Log_norm(csv['minimum_nights'])
    
    last_view=Log_norm(csv['last_review'])
    reviews_per_month=Log_norm(csv['reviews_per_month'])
    price=np.array(csv['price']).reshape(-1,1)
    
    csv['reviews_per_month']=Fillna_with_Min(csv['reviews_per_month'])
    reviews_per_month=Log_norm(csv['reviews_per_month'])
    price=np.array(csv['price']).reshape(-1,1)

    
    data=np.hstack((Group,Roomtype,latitude,longitude,nights,number_of_reviews,host_listings_count,availability,last_view,reviews_per_month,price))
    dtdf=pd.DataFrame(data)
    
    
    ''' 
    Split price, training and testing sets.
    Normalize training data to Euclidean-norm=1 to simplify lasso regression.
    Normalize the test data with same norms as above
    '''
    Y = dtdf.iloc[:, -1].copy()
    X = dtdf.iloc[:,:-2].copy()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size= 0.1, shuffle= True)
    Y_train = np.log(Y_train.to_numpy() + 1)
    Y_test = np.log(Y_test.to_numpy() + 1)

    X_train,norms = normalize(X_train.to_numpy())
    X_test = X_test.to_numpy()/norms

    return  X_train, X_test, Y_train.reshape(-1,1), Y_test.reshape(-1,1)
    
class Lasso:
    
    def __init__(self, lam = 0.01):
        self.lam=lam
        self.epochs=400
        self.train_error = []
        self.val_error = []

    
    def ST(self, rho):
        ''' Soft thresholding function '''
        if rho < -self.lam:
            theta = rho + self.lam
        elif rho > self.lam:
            theta = rho - self.lam
        else:
            theta = 0
        return theta
    
    def rho(self, i, theta, train, price):
        ''' Calculate rho '''
        price_pred=self.predict( train)
        train_i=train[:,i].reshape(-1,1)
        u=theta[i]*train_i
        r=price-price_pred+u
        r=np.dot(train_i.T, r)
        return r
        
    def fit(self, train, price):
        ''' train theta '''
        self.theta=np.ones((train.shape[1], 1))
        for e in range(self.epochs):
            for i in range(train.shape[1]):
                r=self.rho(i, self.theta, train, price)
                self.theta[i]=self.ST(r)

    
    def predict(self, data):
        ''' Predict price on test data, remove normalization'''
        # global prnorm
        prediction=np.dot(data,self.theta) #* prnorm
        return prediction
        
    def error(self, test_data, test_label):
        ''' Calculate MSE. Remove normalization'''
        return mean_squared_error(test_label, self.predict(test_data))
        



'''
Read me: to use this algorithm, you must load/process the data with method Load_Data.
Then you can use the class Lasso. Lasso.fit takes arguments of training dataset and price as np arrays 
and trains the regression coefficients lasso.theta. Lasso.predict takes arguments of test dataset as np array
and calculates/returns predicted price lasso.prediction. Lasso.error takes arg price of test set and returns
MSE. Sample block of code shown below
'''
if __name__ == '__main__':
    path = "CleanedData.csv"
    
    X_train, X_test, Y_train, Y_test = Load_Data(path)
    lasso = Lasso()
    lasso.fit(X_train, Y_train)
    x = lasso.predict(X_test)
    print(lasso.error(X_test, Y_test))

    # print(lasso.train_error)

    '''
    path = "CleanedData.csv"
    Load_Data(path)

    
    '''
