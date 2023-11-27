import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle

class BiLSTM(object):
    def __init__(self, weight='./model/BiLSTM.h5', pattern=''):
        self.model = load_model(weight)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.pattern = pattern

    def predict(self, input):
        flag = True
        if flag:
            # 预测后五分钟的处理逻辑
            if self.pattern == 'olj':
                output = self.model(self.scaler.fit_transform(input)[None])  # input:n*30*3
                res_output = np.reshape(output, (output.shape[0], 5, 3))
                return self.scaler.inverse_transform(res_output[0])
            elif self.pattern == 'temp':
                output = self.model(self.scaler.fit_transform(input)[None])  # input:n*30*8 => n*1*5
                res_output = np.reshape(output, (output.shape[0], 5, 1))  # n*1*5 => n*5*1
                res2inv = np.c_[np.zeros((res_output.shape[0], res_output.shape[1], 7)), res_output]
                return self.scaler.inverse_transform(res2inv[0])
            else:
                return [[]]
        else:
            # 预测后一分钟的处理逻辑
            if self.pattern == 'olj':
                return self.scaler.inverse_transform(self.model(self.scaler.fit_transform(input)[None]))
            elif self.pattern == 'temp':
                data = self.model(self.scaler.fit_transform(input)[None])
                reshape_data = np.zeros((1, 8))
                reshape_data[:, -1] = data
                return self.scaler.inverse_transform(reshape_data)
            else:
                return [[]]


class IForest(object):
    def __init__(self, weight='./model/IForest.pkl'):
        with open(weight, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, input:pd.DataFrame()):
        for col in input.columns:
            field = '{}_anomaly'.format(str(col))
            input[field] = self.model.predict(input[[col]])
