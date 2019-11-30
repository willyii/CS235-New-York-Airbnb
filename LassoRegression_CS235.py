#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
        
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
    samplesize = round(len(dtdf.index)/10)
    test = dtdf.sample(samplesize)
    train = dtdf.drop(test.index)
    price=np.log(train.iloc[:,-1]+1)
    train.drop(train.columns[-1], axis = 1, inplace = True)
    test_pr=np.log(test.iloc[:,-1]+1)
    test.drop(test.columns[-1], axis = 1, inplace = True)
    test=test.to_numpy()
    test_pr=(test_pr.to_numpy()).reshape(-1,1)
    train=train.to_numpy()
    price=(price.to_numpy()).reshape(-1,1)
    train,norms=normalize(train)
    price,prnorm=normalize(price)
    test=test/norms
    samples, feature_space=train.shape
    
class Lasso:
    
    def __init__(self):
        self.lam=0.01
        self.epochs=400
        self.theta=np.ones((feature_space, 1))

        
    
    def predicted_price(self,theta, train):
        ''' Predict price at current theta '''
        predicted_price=np.dot(train,theta)
        return predicted_price
    
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
        price_pred=self.predicted_price(theta, train)
        train_i=train[:,i].reshape(-1,1)
        u=theta[i]*train_i
        r=price-price_pred+u
        r=np.dot(train_i.T, r)
        return r
        
    def fit(self, train, price):
        ''' train theta '''
        for e in range(self.epochs):
            for i in range(feature_space):
                r=self.rho(i, self.theta, train, price)
                self.theta[i]=self.ST(r)        
    
    def predict(self, test):
        ''' Predict price on test data, remove normalization'''
        global prnorm
        self.prediction=prnorm*np.dot(test,self.theta)
        return self.prediction
        
    def error(self, test_price):
        ''' Calculate MSE. Remove normalization'''
        return mean_squared_error(test_price, self.prediction)
        


# In[ ]:





# In[110]:


'''
Read me: to use this algorithm, you must load/process the data with method Load_Data.
Then you can use the class Lasso. Lasso.fit takes arguments of training dataset and price as np arrays 
and trains the regression coefficients lasso.theta. Lasso.predict takes arguments of test dataset as np array
and calculates/returns predicted price lasso.prediction. Lasso.error takes arg price of test set and returns
MSE. Sample block of code shown below
'''
if __name__ == '__main__':
    path = "Data/CleanedData.csv"
    
    Load_Data(path)
    
    '''
    path = "CleanedData.csv"
    Load_Data(path)

    lasso=Lasso()
    lasso.fit(train, price)
    lasso.predict(test)
    lasso.error(test_pr)
    '''
