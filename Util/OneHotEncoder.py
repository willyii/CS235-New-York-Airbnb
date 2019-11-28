import numpy as np
import pandas as pd

class OneHotEncoder:

    def __init__(self):
        self.char_to_vector = {}
        self.char_to_int = {}
        self.num_features = 0

    def fit(self, data: pd.DataFrame,  column_name: str):
        """
        Fitting the training data's specific column. \
        :param data: The original data : Dataframe format.
        :param column_name: The column need to be fit.
        https://blog.csdn.net/gdh756462786/article/details/79161525
        """
        unique_data = data[column_name].unique().tolist()
        self.num_features = len(unique_data)
        base_vector = [0] * self.num_features

        self.char_to_int = dict((c, i) for i, c in enumerate(unique_data))
        for value in self.char_to_int.keys():
            current_vector = base_vector.copy()
            current_vector[self.char_to_int[value]] = 1
            self.char_to_vector[value] = current_vector


    def transform(self, data: pd.DataFrame, column_name: str):
        """
        Transform the training data's specific column. return are numpy.ndarray. which represent the transformed data
        :param data: The original data : Dataframe format.
        :param column_name: The column need to be fit.
        https://blog.csdn.net/gdh756462786/article/details/79161525
        """
        encoded_data  = []
        base_vector = [0] * self.num_features
        for value in data[column_name]:
            if value not in self.char_to_vector.keys():
                encoded_data.append(base_vector)
            else:
                encoded_data.append(self.char_to_vector[value])

        return np.array(encoded_data)



if __name__ == '__main__':

    test = pd.read_csv("AB_NYC_2019.csv")

    test_encoder = OneHotEncoder()
    test_encoder.fit(test, "room_type")
    a  = test_encoder.transform(test, "room_type")
    print(a)
