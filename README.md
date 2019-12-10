# CS235-New-York-Airbnb

This project is made for CS235 course project. It used the dataset of Airbnb dataset from kaggle: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data 
to train model to predict the price of the house in New York.

We tried five algorithms: Lasso(Osten), Neural Network(Dekang), KNN(Xiangting), Randomforest(Faisal) and XGBoost(Xinlong)


### Project Structure

.  
├── Data : dataset files  
│   ├── AB_NYC_2019.csv  
│   ├── CleanedData.csv  
│   └── data_enhanced.csv  
├── KNN:    
│   ├── CleanedData.csv  
│   ├── KNN_cs235.py  
│   ├── KNN_demo.py  
│   └── knn_price_prediction.ipynb  
├── LassoRegression  
│   ├── LassoRegreesion_demo.py  
│   ├── LassoRegression_CS235.py  
├── Neural_Network  
│   ├── Data_Processing.py  
│   ├── NN_CS235.py  
│   ├── checkpoint  
│   ├── data.npy    
│   ├── image.pdf  
│   ├── model.data-00000-of-00001  
│   └── model.index  
├── Notebook: Some Notebook for preprocessing part  
│   ├── Clean\ Data.ipynb  
│   ├── Data\ Explore.ipynb  
│   ├── New_York_City_.png  
│   ├── Poster_result-2.ipynb  
│   ├── Preview_data.ipynb  
│   ├── Room_type\ and\ price.png  
│   ├── Untitled.ipynb  
│   ├── crawler.py  
│   ├── price.png  
│   └── process_availability.ipynb  
├── README.md  
├── RrandomForests  
│   ├── Random_Forest_demo.py  
│   ├── Random_Forests_CS235.py  
├── Util: Some Utils made for preprocess  
│   ├── AB_NYC_2019.csv  
│   ├── FillNa.py   
│   ├── Log1p_Norm.py  
│   ├── Log_Norm.py  
│   ├── Max_Min_Norm.py  
│   ├── OneHotEncoder.py  
│   ├── Z_Score_Norm.py  
│   ├── __init__.py  
├── XGBoost  
│   ├── XGBoost_CS235.py  
│   ├── XGBoost_demo.py  
└── tree.text



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
