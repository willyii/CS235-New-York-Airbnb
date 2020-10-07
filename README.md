# CS235-New-York-Airbnb

This project is made for CS235 course project. It used the dataset of Airbnb dataset from kaggle: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data 
to train model to predict the price of the house in New York.

We tried five algorithms: Lasso(Osten), Neural Network(Dekang), KNN(Xiangting), Randomforest(Faisal) and XGBoost(Xinlong)



### How to run

We prepared the toy datasets for algorithms. For each file end with _demo, we generate the random dataset, and the label is the sum of all features plus noise. All of algorithms are performance well in this toy dataset.   

Use following command to run the XGBoost's demo:

    python3 XGBoost/XGBoost_demo.py 

Use following command to run the Random Forest's demo:  

    python3 RandomFOrests/Random_Forest_demo.py
    
Use following command to run the Neural Network:

    python3 Neural_Network/NN_CS235.py
    
Use following command to run the Lasso's demo:

    python3 LassoRegression/LassoRegression_demo.py
    
Use following command to run the KNN's demo:

    python3 KNN/KNN_demo.py
