# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:15:30 2019

@author: Goktug
"""

# Importing the libraries
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt #Visualization
import pandas as pd # Data Processing
import math
from sklearn.preprocessing import MinMaxScaler #Feature Scaling(Normalization)
from sklearn.metrics import mean_squared_error # Loss Function
from tensorflow.keras.models import Sequential #Create Model
from tensorflow.keras.layers import Dense #Neurons
from tensorflow.keras.layers import LSTM #Long Short Term Memory


import warnings
warnings.filterwarnings('ignore')

# Importing the training set
data = pd.read_csv('C:/Users/Alfredo/Desktop/prices-split-adjusted.csv', index_col = 0)
# First look for dataset
data.info()
data.head()
data.tail()
data.describe()
symbols = list(set(data.symbol))
len(symbols)

plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);
plt.plot(data[data.symbol == 'AMZN'].open.values, color='red', label='open')
plt.plot(data[data.symbol == 'AMZN'].close.values, color='green', label='close')
plt.plot(data[data.symbol == 'AMZN'].low.values, color='blue', label='low')
plt.plot(data[data.symbol == 'AMZN'].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')

plt.figure(figsize=(15, 5));
plt.subplot(1,2,2);
plt.plot(data[data.symbol == 'AMZN'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best');

# Amazon Data Fetching
amazon = data[data['symbol']=='AMZN']
amazon_stock_prices = amazon.close.values.astype('float32')
amazon_stock_prices = amazon_stock_prices.reshape(1762, 1)
amazon_stock_prices.shape

# Feature Scaling
scaler = MinMaxScaler(feature_range = (0, 1))
amazonData = scaler.fit_transform(amazon_stock_prices)
amazonData
plt.plot(amazonData)
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Values')
plt.ylabel('Number of Values')
plt.show()

#Create Train and Test Data
train_size = int(len(amazonData) * 0.80)
test_size = len(amazonData) - train_size
train, test = amazonData[0:train_size,:], amazonData[train_size:len(amazonData),:]
print(len(train), len(test))

x_train = []
y_train = []
timestemps = 10
for i in range(len(train)-timestemps-1):
    a = train[i:(i+timestemps), 0]
    x_train.append(a)
    y_train.append(train[i + timestemps, 0]) 

x_test = []
y_test = []
for i in range(len(test)-timestemps-1):
    a = test[i:(i+timestemps), 0]
    x_test.append(a)
    y_test.append(test[i + timestemps, 0])

#Converting to Numpy Array    
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test) , np.array(y_test)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1])) # For Keras
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])) # For Keras
print(x_train.shape)
print(x_test.shape)

#MODELING by LSTM

#Initializing LSTM
modelLSTM = Sequential()
#Adding a LSTM LAYER
modelLSTM.add(LSTM(10, input_shape = (1, timestemps)))
#Adding Output Layer
modelLSTM.add(Dense(1)) # Output Layer
#Compiling the LSTM
modelLSTM.compile(loss = "mean_squared_error", optimizer = "adam")
#Fitting the LSTM to the Training set
modelLSTM.fit(x_train, y_train, epochs = 200, batch_size = 50)

#Prediction

train_predict = modelLSTM.predict(x_train)
test_predict = modelLSTM.predict(x_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

train_score = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0])) # mean_squared_error -> Loss Function
print("Train Score : %2.f RMSE" % (train_score))
test_score = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print("Test Score : %2.f RMSE" % (test_score))

# Visualising the results
train_predict_plot_LSTM = np.empty_like(amazonData)
train_predict_plot_LSTM[:,:] = np.nan
train_predict_plot_LSTM[timestemps:len(train_predict)+timestemps, :] = train_predict

test_predict_plot_LSTM = np.empty_like(amazonData)
test_predict_plot_LSTM[:, :] = np.nan
test_predict_plot_LSTM[len(train_predict)+(timestemps*2)+1:len(amazonData)-1, :] = test_predict

plt.plot(scaler.inverse_transform(amazonData),color = "red",label = "Real")
plt.plot(train_predict_plot_LSTM,label = "Train Predict",color = "yellow",alpha = 0.7)
plt.plot(test_predict_plot_LSTM,label = "Test Predict",color = "green", alpha = 0.7)
plt.legend()
plt.show()
