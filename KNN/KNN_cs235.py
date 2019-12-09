"""
author: Xiangting Liu
All of the code are made according to the cs235 slides: supervised learning's KNN part
Just use Euclidean distance
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from Util.FillNa import FillNa
from Util.Max_Min_Norm import Max_Min_Norm
from Util.OneHotEncoder import OneHotEncoder

from tqdm import tqdm


class KNN:

    def __init__(self , n_neighbour = 5):
        self.n_neighbour = n_neighbour

    def fit(self, X_train, Y_train):
        """
        Save the training data for the prediction part
        :param X_train:  Trainset, numpy format or Dataframe format
        :param Y_train:  Label of trainset, numpy format or Dataframe format
        :return:
        """
        if type(X_train) == pd.DataFrame:
            self.X_train = X_train.to_numpy()
        else: self.X_train = X_train

        if type(Y_train) == pd.DataFrame:
            self.Y_train = Y_train.to_numpy()
        else:
            self.Y_train = Y_train

    def predict(self, X_test: np.ndarray):
        prediction = []
        for data in tqdm(X_test):
            data_distance = np.subtract(data, self.X_train) # x2 - x1
            data_distance = np.square(data_distance) #(x2 - x1)^2
            data_distance = np.sum(data_distance, axis = 1).reshape((-1,1)) # (x2 - x1)^2 + (y2 - y1)^2
            Y_train_tmp = self.Y_train.copy()

            # data_distance_label = np.hstack((data_distance, self.Y_train))
            # close_data_index = data_distance.argmin( axis = 0 )
            tmp = []
            for i in range(self.n_neighbour):
                index = np.argmin(data_distance, axis=0)
                tmp.append(Y_train_tmp[index])
                data_distance = np.delete(data_distance, index)
                Y_train_tmp = np.delete(Y_train_tmp, index)
            tmp[0] = tmp[0][0]
            prediction.append(np.mean(tmp))
        return prediction


    def Loss(self, predicted, actual):
        return mean_squared_error(predicted, actual)






def Load_Data(path):
    """
    This function aim to load data and preprocess the features
    ['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
       'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
       'minimum_nights', 'number_of_reviews', 'last_review',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']
    :param path:  the path to the dataset
    :return: processed dataset: numpy array format
    """
    raw_data = pd.read_csv(path)

    raw_label = raw_data[["price"]].copy()
    raw_data = raw_data.drop(columns= ["price"])
    X_train, X_test, y_train, y_test = train_test_split(raw_data, raw_label, test_size = 0.1, random_state = 42,shuffle= True)



    # process the "neighbourhood_group"
    group_encoder = OneHotEncoder()
    group_encoder.fit(X_train, "neighbourhood_group")

    # process the "neighbourhood"
    neighbour_encoder = OneHotEncoder()
    neighbour_encoder.fit(X_train, "neighbourhood")

    # process the latitude
    latitude_norm = Max_Min_Norm()
    latitude_norm.fit(X_train, "latitude")

    # process the longitude
    longitude_norm = Max_Min_Norm()
    longitude_norm.fit(X_train, "longitude")

    # process the room_type
    room_type_encoder = OneHotEncoder()
    room_type_encoder.fit(X_train, "room_type")

    # process the review_per_month
    review_per_month = FillNa()
    review_per_month.fit(X_train, "reviews_per_month", "mean")


    group_train = group_encoder.transform(X_train, "neighbourhood_group")
    neighbourhood_train = neighbour_encoder.transform(X_train, "neighbourhood")
    latitude_train = latitude_norm.transform(X_train, "latitude")
    longitude_train = longitude_norm.transform(X_train,"longitude")
    room_type_train = room_type_encoder.transform(X_train, "room_type")
    minimum_nights_train = np.log1p(X_train[["minimum_nights"]]).to_numpy()
    number_of_reviews_train = X_train[["number_of_reviews"]].to_numpy()
    last_review_train = X_train[["last_review"]].to_numpy()
    reviews_per_month_train = review_per_month.transform(X_train, "reviews_per_month")
    calculated_host_listings_count_train = X_train[["calculated_host_listings_count"]].to_numpy()
    availability_365_train = X_train[["availability_365"]].to_numpy()
    rate_train = X_train[["rate"]].to_numpy()

    X_train = np.hstack((group_train, neighbourhood_train, latitude_train,longitude_train, room_type_train, minimum_nights_train, number_of_reviews_train ,
                         last_review_train,reviews_per_month_train, calculated_host_listings_count_train, availability_365_train, rate_train ))
    Y_train = np.log1p(y_train).to_numpy()

    group_test = group_encoder.transform(X_test, "neighbourhood_group")
    neighbourhood_test = neighbour_encoder.transform(X_test, "neighbourhood")
    latitude_test = latitude_norm.transform(X_test, "latitude")
    longitude_test = longitude_norm.transform(X_test, "longitude")
    room_type_test = room_type_encoder.transform(X_test, "room_type")
    minimum_nights_test = np.log1p(X_test[["minimum_nights"]]).to_numpy()
    number_of_reviews_test = X_test[["number_of_reviews"]].to_numpy()
    last_review_test = X_test[["last_review"]].to_numpy()
    reviews_per_month_test = review_per_month.transform(X_test, "reviews_per_month")
    calculated_host_listings_count_test = X_test[["calculated_host_listings_count"]].to_numpy()
    availability_365_test = X_test[["availability_365"]].to_numpy()
    rate_test = X_test[["rate"]].to_numpy()

    X_test = np.hstack((group_test, neighbourhood_test, latitude_test, longitude_test, room_type_test,
                         minimum_nights_test, number_of_reviews_test,
                         last_review_test, reviews_per_month_test, calculated_host_listings_count_test,
                         availability_365_test, rate_test))
    Y_test = np.log1p(y_test).to_numpy()

    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


if __name__ == '__main__':

    print("running...")
    path = "CleanedData.csv"
    X_train, Y_train, X_test, Y_test = Load_Data(path)
    knn = KNN()
    knn.fit(X_train, Y_train)
    kk = knn.predict(X_test[:100,:])
    print(knn.Loss(kk, Y_test[:100]))
