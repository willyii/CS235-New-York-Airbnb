import Data_Processing
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class NN(object):
    
    def __init__(self, url):

        self.process = Data_Processing.Data_processing(url)

    def clean_data(self):

        # clean for price
        self.process.data['Log_price'] = np.log1p(self.process.data['price'])
        self.process.data['zscore_price'] = self.process.Z_Score('Log_price')
        self.process.data = self.process.data[np.abs(self.process.data['zscore_price']) <= 3]
        self.process.data = self.process.data.drop(columns=['Log_price'])
        self.process.data = self.process.data.drop(columns=['zscore_price'])

        # clean for minimum_Nights method form Group member Xinlong Yi
        self.process.data['log_minimum_nights'] = np.log(self.process.data["minimum_nights"])  # log nigths

        # plots
        plot_info = self.process.data.boxplot(column='log_minimum_nights', return_type='dict')

        # Delete outlier
        threshold = plot_info['whiskers'][1].get_ydata()[1]
        self.process.data = self.process.data[self.process.data['log_minimum_nights'] <= threshold]
        # print("There are " , price_deleted["log_nights"].count() - mini_nights_deleted["log_nights"].count() , "outliers")
        self.process.data = self.process.data.drop(columns=['log_minimum_nights'])
        self.process.data['last_review'] = self.process.Fillna_with_Zero('last_review')
        self.process.data['reviews_per_month'] = self.process.Fillna_with_Zero('reviews_per_month')

    def Normalization(self):
        #Normalization for CSV, then save as .npy file
        # One_Hot_encoding for group, neibourhood and room type
        Group = self.process.OneHotEncoding('neighbourhood_group')
        Neibourhood = self.process.OneHotEncoding('neighbourhood')
        Roomtype = self.process.OneHotEncoding('room_type')

        #max-min norm
        latitude = self.process.Max_min_norm('latitude')
        longitude = self.process.Max_min_norm('longitude')
        host_listings_count = self.process.Max_min_norm('calculated_host_listings_count')
        availability = self.process.Max_min_norm('availability_365')

        # log_norm
        number_of_reviews = self.process.Log_norm('number_of_reviews')
        nights = self.process.Log_norm('minimum_nights')
        # processing for last_review
        self.process.data['last_review'] = pd.to_datetime(self.process.data['last_review'])
        self.process.data['last_review'] = (
                    self.process.data['last_review'].max() - self.process.data['last_review']).dt.days
        last_view = self.process.Log_norm('last_review')

        # processing for reviews_pre_month

        reviews_per_month = self.process.Log_norm('reviews_per_month')

        price = self.process.Log_1pNorm('price')
        print(price)
        data = np.hstack(
            (Group, Neibourhood, Roomtype, latitude, longitude, nights, number_of_reviews, host_listings_count,
             availability, last_view, reviews_per_month, price))
        print(data.shape)
        np.save('data.npy', data)

    def spilt_dataset(self, dataset, rate):
        #spilt dataset according to rate%
        np.random.shuffle(dataset)
        spilt_length = int(dataset.shape[0] * rate)
        train_example = dataset[0:spilt_length, 0:-1]
        train_labels = dataset[0:spilt_length, -1]
        test_example = dataset[spilt_length:, 0:-1]
        test_labels = dataset[spilt_length:, -1]
        return train_example, train_labels, test_example, test_labels

    def spilt_labels(self,dataset):
        #spilt dataset to examples and labels
        #dataset should be type of numpy
        examples = dataset[:, 0:-1]
        labels = dataset[:, -1]
        return examples, labels

    def spilt_to_five(self,dataset):
        #spilt dataset to five fold
        #dataset should be type of numpy
        length = int(dataset.shape[0] * 0.2)
        A = dataset[0:length, :]
        B = dataset[length:length * 2, :]
        C = dataset[length * 2:length * 3, :]
        D = dataset[length * 3:length * 4, :]
        E = dataset[length * 4:, :]
        return A, B, C, D, E

    def train_model(self, model_name, data_length):
        # 5 model to compare
        #https: // www.tensorflow.org / tutorials / keras / regression
        if model_name == 'A':
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=(data_length,)),
                layers.Dense(64, activation='relu'),
                layers.Dense(16, activation='linear'),
                layers.Dense(1)
            ])
        elif model_name == 'B':
            model = keras.Sequential([
                layers.Dense(32, activation='linear', input_shape=(data_length,)),
                layers.Dense(16, activation='linear'),
                layers.Dense(8, activation='linear'),
                layers.Dense(1)
            ])
        elif model_name == 'C':
            model = keras.Sequential([
                layers.Dense(32, activation='relu', input_shape=(data_length,)),
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(1)
            ])
        elif model_name == 'D':
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(data_length,)),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(1)
            ])
        elif model_name == 'E':
            model = keras.Sequential([
                layers.Dense(64, activation='linear', input_shape=(data_length,)),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='linear'),
                layers.Dense(8, activation='relu'),
                layers.Dense(1)
            ])

        else:
            model = keras.Sequential([
                layers.Dense(1, activation='linear', input_shape=(data_length,)),

            ])
        optimizer = tf.keras.optimizers.Adam(0.0001)
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    def plot_history(self,history):
        #draw a history plot when training
        #https: // www.tensorflow.org / tutorials / keras / regression

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

    def train(self,Data_url):
        #Data_url is the address of Data.npy
        #https: // www.tensorflow.org / tutorials / keras / overfit_and_underfit
        # https: // www.tensorflow.org / tutorials / keras / regression
        #https: // www.tensorflow.org / tutorials / keras / save_and_load
        data = np.load(Data_url)

        train_example, train_labels, test_example, test_labels = self.spilt_dataset(data, 0.9)
        data_length = train_example.shape[1]

        EPOCHS = 1000
        checkpoint_path = ''
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=1000)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=50)
        model = self.train_model('C', data_length)
        model.summary()
        history = model.fit(x=train_example, y=train_labels, epochs=EPOCHS,

                            validation_data=[test_example, test_labels],
                            callbacks=[cp_callback, early_stop])
        self.plot_history(history)
        loss, mae, mse = model.evaluate(x=test_example, y=test_labels)
        model.save_weights('model')
        print(mse)

    def Compare_Network(self,Data_Url):
        #compare 5 model on 5-Fold cross validation
        data = np.load(Data_Url)
        data_A, data_B, data_C, data_D, data_E = self.spilt_to_five(data)
        EPOCHS = 1000
        early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=50)
        model_list = ['A', 'B', 'C', 'D', 'E']
        data_list = [data_A, data_B, data_C, data_D, data_E]
        average_mse_list = []

        for model_name in model_list:
            mse_list = []
            index = 0
            for testset in data_list:
                train_data_list = data_list.copy()
                del train_data_list[index]
                trainset = np.vstack((train_data_list[0], train_data_list[1], train_data_list[2], train_data_list[3]))
                index += 1
                test_example, test_label = self.spilt_labels(testset)
                train_example, train_label = self.spilt_labels(trainset)
                input_length = train_example.shape[1]
                model = self.train_model(model_name, input_length)
                # test_dataset = tf.data.Dataset.from_tensor_slices((dataset[0], train_labels))
                model.summary()
                model.fit(x=train_example, y=train_label, epochs=EPOCHS,

                          validation_data=[test_example, test_label],
                          callbacks=[early_stop])
                loss, mae, mse = model.evaluate(test_example, test_label)
                mse_list.append(mse)
            average_mse_list.append(np.mean(mse_list))

        result = {'mse': average_mse_list,
                  'model_name': model_list}
        result = pd.DataFrame(result)
        plt.figure()
        plt.xlabel('model_name')
        plt.ylabel('average_mse')
        plt.plot(result['model_name'], result['mse'])
        plt.legend()
        plt.savefig('image.pdf')
        plt.show()

    def evulate(self,Data_url):
        # https: // www.tensorflow.org / tutorials / keras / save_and_load
        data = np.load(Data_url)
        train_example, train_labels, test_example, test_labels = self.spilt_dataset(data, 0.9)
        data_length = train_example.shape[1]
        model=self.train_model('C',data_length)
        model.load_weights('model')
        test_predictions=model.predict(test_example)
        test_predictions=np.expm1(test_predictions)
        test_labels=np.expm1(test_labels)
        plt.scatter(test_labels,test_predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    Regressor=NN('../Data/AB_NYC_2019.csv')
    Regressor.clean_data()
    Regressor.Normalization()
    Regressor.train('data.npy')
    Regressor.evulate('data.npy')
