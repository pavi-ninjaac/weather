# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:02:22 2020

@author: ninjaac
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=pd.read_csv(r'C:\Users\ninjaac\Desktop\python\dataAnalysis\wether_RNN\training.csv')
test_data=pd.read_csv(r'C:\Users\ninjaac\Desktop\python\dataAnalysis\wether_RNN\testing.csv')
#split the training data

train_data_2=train_data.temparature.str.split(expand=True)
train_data_3=train_data_2.drop(columns=0)
test_data=test_data.drop(columns=['Date time'])
train_data.keys()
#feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(train_data_3)
X_test = sc_X.transform(test_data)

x_train = []
y_train = []
n_future = 4 # next 4 days temperature forecast
n_past = 30 # Past 30 days 
for i in range(0,len(X_train)-n_past-n_future+1):
    x_train.append(X_train[i : i + n_past , 0])     
    y_train.append(X_train[i + n_past : i + n_past + n_future , 0 ])
x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )

from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout

model=Sequential()

model.add(Bidirectional(LSTM(unit=30,return_sequence=True,input_shape=(x_train.shape[1],1))))
model.add(Dropout(0.2))

model.add(LSTM(unit=30,return_sequence=True ))
model.add(Dropout(0.2))

model.add(LSTM(unit=30,return_sequence=True ))
model.add(Dropout(0.2))

model.add(Dense(units=n_future,activation='linear'))