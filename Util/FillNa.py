import pandas as pd

class FillNa:

    def __init__(self):
        self.fill_value = 0

    def fit(self, data: pd.DataFrame,  column_name: str, method ="mean" ):
        """
        Fill the missing value, default use mean to fill
        :param data: Dataset with missing value. Dataframe format
        :param column_name: The name of column. String format
        :param method: filling method, default "mean", also has [mean, median, mode, max, min] or specific value
        :return the dataframe column with filled value
        """
        filling_name = ["mean", "median", "mode", "max", "min"]
        if method in filling_name:
            if method == "mean":
                self.fill_value = data[column_name].mean()
            elif method == "median":
                self.fill_value = data[column_name].median()
            elif method == "mode":
                self.fill_value = data[column_name].mode()
            elif method == "max":
                self.fill_value = data[column_name].max()
            elif method == "min":
                self.fill_value = data[column_name].min()
        else:
            self.fill_value = method

    def transform(self, data: pd.DataFrame,  column_name: str):
        return data[[column_name]].fillna(self.fill_value).to_numpy()

if __name__ == '__main__':
    test = pd.read_csv("AB_NYC_2019.csv")
    print(test.columns)
    test_encoder = FillNa()
    test["reviews_per_month"] =  test_encoder.fill(test, "reviews_per_month")
    print(test)