import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from Util.OneHotEncoder import OneHotEncoder





# The node of Single Tree
class Node:
    # Inital
    def __init__(self):
        self.left_node = None
        self.right_node = None

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

            sorted_values = np.sort(np.unique(data[[feature]],axis=0), axis=0).flat  # Drop duplicates and sort the values of sepcific feature
            if len(sorted_values) > 1000:
                sorted_values = np.random.choice(sorted_values, 100, replace= False)
            for sample_index in range(len(sorted_values) -1 ):

                split_value = (sorted_values[sample_index] + sorted_values[sample_index + 1]) / 2

                # left_score
                left_label = label[data[feature] <= split_value]
                left_previou_preidct = previous_label[data[feature] <= split_value]
                left_score = self.structure_score(left_label, left_previou_preidct)

                # right_score
                right_label = label[data[feature] > split_value]
                right_previou_preidct = previous_label[data[feature] > split_value]
                right_score = self.structure_score(right_label, right_previou_preidct)

                # Gain
                current_gain = 0.5 * (left_score + right_score - self.score )
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
        return (g.sum() ** 2)/(2 * len(label) )

    def leaf_weight(self, label, previous_label):
        g = 2 * (previous_label.to_numpy() - label.to_numpy())
        h = 2
        return (g.sum()) / (2 * len(label))


# SingleTree in XGBoost
class SingleTree:

    def __init__(self):
        self.max_depth = 6
        self.start_node = Node()

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
        if start_node.split_feature == "" or current_depth > self.max_depth:
            return
        else:
            # inital the sub tree
            start_node.left_node = Node()
            start_node.right_node = Node()

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

    def __init__(self):
        self.eta = 0.3 # step size shrinkage used in update to prevents overfitting.
        self.gamma = 0 # minimum loss reduction required to make a further partition on a leaf node of the tree
        self.max_depth = 2 # maximum depth of a tree
        self.subsample = 1 #subsample ratio of the training instance.
        self.tree_list = [] ##Tree list, store the trees

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

    def fit(self, train_data, train_label):
        """
        This function used for train XGBoost model based on train_data and train_label
        :param train_data: Train data
        :param train_label: Train label
        """
        if(type(train_data) == np.ndarray):
            train_data = self.np_to_df(train_data)

        if self.tree_list == []:
            previous_predict = pd.DataFrame({'predict': [0] * len(train_data)})
            previous_error = self.Loss(train_label, previous_predict)

        while(1):
            # New Tree
            new_tree = SingleTree()
            # Tree Samples
            tree_data = train_data.copy()
            tree_data["label"] = train_label
            tree_data["previous_label"] = previous_predict# pd.concat([train_data, train_label, previous_predict], ignore_index= True, axis= 1 , sort= False)
            tree_train_sub_sample = tree_data.sample(frac=0.7, replace= False, axis= 0 )
            tree_previous_label = tree_train_sub_sample.iloc[:, [-1]]
            tree_label = tree_train_sub_sample.iloc[:,[-2]]
            tree_data = tree_train_sub_sample.iloc[:,:-2]
            tree_train_sub_feature = tree_data.sample(frac=0.7, replace= False, axis= 1 )

            # train the tree
            new_tree.fit(tree_train_sub_feature, tree_label,tree_previous_label)
            self.tree_list.append(new_tree)

            # New tree to predict the train_data
            tree_predict = new_tree.predict(train_data)

            current_predict = tree_predict  + previous_predict
            print(current_predict)
            current_error = self.Loss(train_label, current_predict)

            print("After " + str(len(self.tree_list)) + " trees, the error becomes " + str(current_error))

            if abs(current_error - previous_error) < 0.0001 or len(self.tree_list) > 100:
                break
            else:
                previous_predict = current_predict
                previous_error = current_error

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
            predict = predict + tree.predict(test_data)
        return predict

    def Loss(self, label, predict):
        """
        Compute the loss of current mode
        :param label: actual label
        :param predict: predict label
        :return: the error of current predict
        """

        return mean_squared_error(label.to_numpy(), predict.to_numpy())
        # return mean_squared_error(np.power(np.e, label.to_numpy()), np.power(np.e, predict.to_numpy()))


def Load_Data(path):
    """
    This function aim to load data and preprocess the features
    :param path:  the path to the dataset
    :return: processed dataset: numpy array format
    """
    raw_data = pd.read_csv(path)

    # process the neighbour_hood_group
    group_encoder = OneHotEncoder.OneHotEncoder()
    group_encoder.fit(raw_data, "neighbourhood_group")

    group_encoded = group_encoder.transform(raw_data, "neighbourhood_group")
    print(group_encoded.shape)


if __name__ == '__main__':

    print("I'm busy now")
    path = "Data/AB_NYC_2019.csv"
    Load_Data(path)
    # # A =  np.random.rand(1, 500) * 100
    # # B =  np.random.rand(1, 500)
    # # train_data = pd.DataFrame({"A": A[0], "B": B[0]})
    # # train_label = pd.DataFrame({"A": A[0]*B[0]  + np.random.random_integers(0,100)})
    #
    # train_data = np.load("processed.npy")
    #
    # train_label = pd.read_csv("AB_NYC_2019.csv")[["price"]]
    # train_label =  np.log(train_label+ 1)
    # train_label.rename(columns={"price": "predict"})
    #
    #
    # test = XGBoost()
    # test.fit(train_data, train_label)
    # print(test.predict(train_data))
    #
    #
    # # test = XGBRegressor()
    # # test.fit(train_data, train_label)
    # # print(mean_squared_error(train_label, test.predict(train_data)))