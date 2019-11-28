# CS235-New-York-Airbnb
Here are the code for the CS235 project.

### Update on 27 Nov 2019

Finished Util package. This package include some basic function to preprocess the features. If you want to use that 
pls use following code

    from Util.OneHotEncoder import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit()
    
to import the method you want. Every py file has a small example. Pls read it before use. 

I change the structure of the project. Pls put your algorithm in root. The datasets are in Data fold.

--------------------------------------------
Some points about our algorithm:

- Pls make sure it is a class, contains "fit" and "predict" functions. Where fit function means this model is training. The "predict" function means 
it predicting the label of dataset. Like XGBoost_CS235.py file 

- We use Mean Square Error(MSE) to measure the performace of the model. 

- In your algorithm file, pls contain a main function. Which should contain "preprocess data" -> "split testset and train set (trainset 90%)"" 
-> "train model" -> "predict the testset data" -> "measure the error in testset" 

- We use log(price + 1) as label of our dataset. For other features' preprocess, you can choose by yourself.

