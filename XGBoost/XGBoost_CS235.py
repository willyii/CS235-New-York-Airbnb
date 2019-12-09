"""
# -*- coding:utf-8 -*-
# Author: Xinlong Yi
All of code are made from scrach followed the Chen's Slide: https://homes.cs.washington.edu/~tqchen/data/pdf/BoostedTree.pdf and
the paper: https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from Util.OneHotEncoder import OneHotEncoder
from Util.Log1p_Norm import Log1p_Norm
from Util.Max_Min_Norm import Max_Min_Norm
from Util.FillNa import FillNa
import sys
import warnings




# The node of Single Tree
class Node:
    # Inital
    def __init__(self, min_child_weight):
        self.left_node = None
        self.right_node = None
        self.min_child_weight = min_child_weight

    def exact_search(self, data, label , previous_label):
        """
        This function used for find the best split of every node.
        :param data: array or sparse matrix, shape (n_samples, n_features)
        """

        # If do search, computer the node information
        self.score = self.structure_score(label, previous_label) # construct score
        self.spilt_value = 0# Optimal w_j
        self.leaf = -self.leaf_weight(label, previous_label)
        self.split_feature = ""
        self.gain = 0
        cols = data.columns

        for feature in cols: # every feature
            sorted_values = np.sort(np.unique(data[feature])).flat  # Drop duplicates and sort the values of sepcific feature
            if len(sorted_values) > 100:
                sorted_values = np.random.choice(sorted_values, 10, replace= False)
            for sample_index in range(len(sorted_values) -1 ):

                split_value = (sorted_values[sample_index] + sorted_values[sample_index + 1]) / 2

                # left_score
                left_label = label[data[feature] <= split_value]
                if len(left_label) < self.min_child_weight: # child_weight
                    continue
                left_previou_preidct = previous_label[data[feature] <= split_value]
                left_score = self.structure_score(left_label, left_previou_preidct)

                # right_score
                right_label = label[data[feature] > split_value]
                if len(right_label) < self.min_child_weight:# child_weight
                    continue
                right_previou_preidct = previous_label[data[feature] > split_value]
                right_score = self.structure_score(right_label, right_previou_preidct)

                # Gain
                current_gain = 0.5 * (left_score + right_score - self.score)
                if current_gain > self.gain and current_gain >= 0.5:
                    self.split_feature = feature
                    self.spilt_value = split_value
                    self.gain = current_gain


    def structure_score(self, label, previous_label):
        """
        This function used for compute the structure score of specific Node
        :param label: The label of the dataset in that node
        :param previou_predict: The predicted label of that dataset in last iteration
        :return: The structure score of that Node
        """
        g = 2 * (previous_label.to_numpy() - label.to_numpy())
        h = 2
        r = (g.sum() ** 2)/(2 * len(label) )
        if r == np.nan:
            return -1
        return r


    def leaf_weight(self, label, previous_label):
        g = 2 * (previous_label.to_numpy() - label.to_numpy())
        h = 2
        r = (g.sum()) / (2 * len(label) )
        if r == np.nan:
            return 0
        return r


# SingleTree in XGBoost
class SingleTree:

    def __init__(self, max_depth, min_child_weight):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.start_node = Node( self.min_child_weight)

    def fit(self, train_data , train_label, previous_label):
        """
        This function used for train a single tree in XGboost
        :param train_data: Training data, Dataframe format
        :param train_label: Training label, Dataframe format
        :param previous_label: Prediction in previous interation. Datafram format
        """
        start_depth = 1
        self.construct_node(self.start_node, train_data, train_label, previous_label, start_depth)


    def construct_node(self, start_node:Node, train_data , train_label, previous_label, previous_depth):
        """
        This funtion used to construct the Nodes in Single tree in XGBoost.
        Basic logic: exact_search the best split feature and value. If cannot find or exceed the max depth. Set this node as leaf node
        :param start_node: Current Node of that tree
        :param train_data: Training data, Dataframe format
        :param train_label: Training label, Dataframe format
        :param previous_label: Prediction in previous interation. Datafram format
        :param previous_depth: the level of tree above this node
        """

        current_depth = previous_depth + 1
        # construct the root node
        start_node.exact_search(train_data, train_label, previous_label)

        # If no new split info
        if start_node.split_feature == "" or current_depth > self.max_depth or start_node.gain == 0:
            return
        else:
            # inital the sub tree
            start_node.left_node = Node( self.min_child_weight)
            start_node.right_node = Node( self.min_child_weight)

            # Left subtree
            left_data = train_data[train_data[start_node.split_feature] <= start_node.spilt_value]
            left_label = train_label[train_data[start_node.split_feature] <= start_node.spilt_value]
            left_previou_preidct = previous_label[train_data[start_node.split_feature] <= start_node.spilt_value]

            # Right subtree
            right_data = train_data[train_data[start_node.split_feature] > start_node.spilt_value]
            right_label = train_label[train_data[start_node.split_feature] > start_node.spilt_value]
            right_previou_preidct = previous_label[train_data[start_node.split_feature] > start_node.spilt_value]

            # Construct the subtrees
            self.construct_node(start_node.left_node, left_data, left_label, left_previou_preidct, current_depth)
            self.construct_node(start_node.right_node, right_data, right_label, right_previou_preidct, current_depth)


    def predict(self, test_data):
        """
        Predict the label of test_data.
        :param test_data: Dataset that need to be predict
        :return: the label of test_data. Dataframe format
        """
        test_data.index = range(len(test_data))
        predicted_data = self.predict_node(self.start_node,test_data)
        return predicted_data.sort_index()[["predict"]]

    def predict_node(self, current_node: Node, test_data):
        """
        This function used for act a predict in single node
        :param current_node: currrent Node: Node format
        :param test_data: data that split to this node
        :return: predicted data
        """
        if current_node.left_node == None and current_node.right_node == None: # leaf node
            test_data["predict"] =  current_node.leaf
            return test_data
        else:
            left_data = test_data[test_data[current_node.split_feature] <= current_node.spilt_value]
            right_data = test_data[test_data[current_node.split_feature] > current_node.spilt_value]
            left_predict = self.predict_node(current_node.left_node, left_data)
            right_predict = self.predict_node(current_node.right_node, right_data)
            combined_data = left_predict.append(right_predict)
            return combined_data


class XGBoost:

    def __init__(self, paras:dict ):
        self.learning_rate = 0.3 # step size shrinkage used in update to prevents overfitting.
        self.max_depth = 4 # maximum depth of a tree
        self.subsample = 0.6 #subsample ratio of the training instance.
        self.colsample = 0.6#subsample ratioe of the columns.
        self.tree_list = [] ##Tree list, store the trees.
        self.early_stop_round = 5 #Early Stop
        self.n_estimator = 1000 # number of estimator
        self.val_errors = []
        self.error_threadhold = 0.0001 #
        self.train_errors = []
        self.min_child_weight = 10
        self.parse_para(paras)

    def parse_para(self, para: dict):
        """
        parse the parameters of xgboost model
        :param para: parameters, dictionary format.
        """
        for key in para.keys():
            if key == "learning_rate":
                self.learning_rate = para[key]
                continue
            if key == "max_depth":
                self.max_depth = para[key]
                continue
            if key == "subsample":
                self.subsample = para[key]
                continue
            if key == "colsample":
                self.colsample = para[key]
                continue
            if key == "early_stop_round":
                self.early_stop_round = para[key]
                continue
            if key == "n_estimator":
                self.n_estimator = para[key]
                continue
            if key == "min_child_weight":
                self.min_child_weight = para[key]
                continue


    def np_to_df(self, train_data):
        """
        This function used for trans the data from numpy to pd.dataframe
        :param train_data: train _data, numpy format
        :return: train_data, dataframe format
        """
        feature_num = train_data.shape[1]

        col_names = []
        for i in range(feature_num):
            col_names.append("feature_" + str(i))

        return pd.DataFrame(train_data, columns=col_names)

    def fit(self, train_data, train_label, test_data = None, test_label = None):
        """
        This function used for train XGBoost model based on train_data and train_label
        :param train_data: Train data
        :param train_label: Train label
        """
        if(type(train_data) == np.ndarray):
            train_data = self.np_to_df(train_data)
        if(type(train_label) == np.ndarray):
            train_label = self.np_to_df(train_label)

        if (type(test_data) == np.ndarray) and (type(test_label) == np.ndarray):
            test_data = self.np_to_df(test_data)
            test_label = self.np_to_df(test_label)
        else:
            self.early_stop_round = self.n_estimator


        if self.tree_list == []:
            previous_predict = pd.DataFrame({'predict': [0] * len(train_data)})
            previous_error = self.Loss(train_label, previous_predict)

        while(1):
            print("Now training the No.{} tree".format(len(self.tree_list) + 1))
            # New Tree
            new_tree = SingleTree(self.max_depth, self.min_child_weight)
            # Tree Samples
            tree_data = train_data.copy()
            tree_data["label"] = train_label
            tree_data["previous_label"] = previous_predict# pd.concat([train_data, train_label, previous_predict], ignore_index= True, axis= 1 , sort= False)
            tree_train_sub_sample = tree_data.sample(frac=self.subsample, replace= False, axis= 0 )
            tree_previous_label = tree_train_sub_sample.iloc[:, [-1]]
            tree_label = tree_train_sub_sample.iloc[:,[-2]]
            tree_data = tree_train_sub_sample.iloc[:,:-2]
            tree_train_sub_feature = tree_data.sample(frac=self.colsample, replace= False, axis= 1 )

            # train the tree
            new_tree.fit(tree_train_sub_feature, tree_label,tree_previous_label)
            self.tree_list.append(new_tree)

            # New tree to predict the train_data
            tree_predict = new_tree.predict(train_data)

            current_predict = self.learning_rate * tree_predict  + previous_predict
            current_error = self.Loss(train_label, current_predict)

            print("After " + str(len(self.tree_list)) + " trees, the error becomes " + str(current_error))
            self.train_errors.append(current_error)
            if type(test_data) == pd.DataFrame  and type(test_label) == pd.DataFrame:
                self.validation(test_data, test_label)

            if len(self.tree_list) > self.n_estimator or self.check_early_stop() or (previous_error - current_error) < self.error_threadhold:
                break
            else:
                previous_predict = current_predict
                previous_error = current_error

    def check_early_stop(self):
        if len(self.val_errors) < self.early_stop_round or self.early_stop_round == self.n_estimator:
            return False
        for i in range(self.early_stop_round - 1):
            if self.val_errors[-(i+1)] < self.val_errors[-(i+2)]:
                return False
        return True


    def predict(self, test_data):
        """
        This function used for predict the label of data
        :param test_data: Test data
        :return: predict label: dataframe type with column named "predict"
        """

        if(type(test_data) == np.ndarray):
            test_data = self.np_to_df(test_data)
        predict = pd.DataFrame({'predict': [0] * len(test_data)})
        for tree in self.tree_list:
            predict =  predict + self.learning_rate * tree.predict(test_data)
        return predict

    def Loss(self, label, predict):
        """
        Compute the loss of current mode
        :param label: actual label
        :param predict: predict label
        :return: the error of current predict
        """
        return mean_squared_error(label.to_numpy(), predict.to_numpy())

    def validation(self, test_data, test_label):
        test_predict = self.predict(test_data)
        self.val_errors.append(self.Loss(test_label, test_predict))
        print("Performance on test set is {}".format(self.val_errors[-1]))


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

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    path = "CleanedData.csv"
    X_train, Y_train, X_test, Y_test = Load_Data(path)

    paras = {}
    paras["learning_rate"] = 0.3
    paras["max_depth"] = 4
    paras["subsample"]  = 0.6
    paras["colsample"] = 0.5
    paras["early_stop_round"] = 5
    paras["n_estimator"] = 10
    paras["min_child_weight"] = 10

    test = XGBoost(paras)
    test.fit(X_train, Y_train, X_test, Y_test)
    print(test.train_errors)
    print(test.val_errors)



    # from xgboost import XGBRegressor
    # xgb = XGBRegressor(n_estimators= 1000, learning_rate=0.3)
    # xgb.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], early_stopping_rounds=5, verbose= True, eval_metric="rmse")