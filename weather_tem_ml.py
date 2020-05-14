# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:13:00 2020

@author: ninjaac
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv(r'C:\Users\ninjaac\Desktop\python\dataAnalysis\wether_RNN\weahterML\weatherHistory.csv\weatherHistory.csv')

dataset.keys()
data=dataset.drop(columns=['Formatted Date','Summary','Apparent Temperature (C)', 'Wind Bearing (degrees)', 'Loud Cover','Daily Summary'])

tem_data=data.iloc[0:70000,1:2].values
#scaling the input
from sklearn.preprocessing import MinMaxScaler
sca=MinMaxScaler()
tem_data=sca.fit_transform(tem_data)


#train test split
from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(tem_data,test_size=0.25,random_state=42)

#train test ceration
X_train = []
y_train = []
n_future = 1 # next 4 days temperature forecast
n_past = 30 # Past 30 days 
for i in range(0,len(x_train)-n_past-n_future+1):
    X_train.append(x_train[i : i + n_past , 0])     
    y_train.append(x_train[i + n_past : i + n_past + n_future , 0 ])
X_train , y_train = np.array(X_train), np.array(y_train)

#test set creation
X_test = []
y_test = []
n_future = 1 # next 4 days temperature forecast
n_past = 30 # Past 30 days 
for i in range(0,len(x_test)-n_past-n_future+1):
    X_test.append(x_test[i : i + n_past , 0])     
    y_test.append(x_test[i + n_past : i + n_past + n_future , 0 ])
X_test , y_test = np.array(X_test), np.array(y_test)
#converting matrix to a vector
y_test=y_test[:,0]
y_train=y_train[:,0]

#train the model
from sklearn.ensemble import RandomForestRegressor

ran=RandomForestRegressor(n_estimators=5,n_jobs=-1,random_state=42)
ran.fit(X_train,y_train)
predicted_tem=ran.predict(X_test)

#ploting the output
plt.plot(y_test,color='red')
plt.plot(predicted_tem,color='blue')
plt.xlim((0,150))

plt.show()

#train decision ttree regressor
from sklearn.tree import DecisionTreeRegressor
den=DecisionTreeRegressor(max_depth=80,random_state=0)
den.fit(X_train,y_train)
pre_tem=den.predict(X_test)


plt.plot(y_test,color='red')
plt.plot(pre_tem,color='blue')
plt.xlim((0,50))

plt.show()



