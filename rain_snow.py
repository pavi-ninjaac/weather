# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:59:44 2020

@author: ninjaac
"""
"""
TOTAL REPORT ""on rain or snow day classification""
totally used 3 model
model                   accuracy
RandomForestClassifier: 0.9637675116744496

SCV with grid search  : 0.9988603513453413

KNN                   : 0.9800283522348232

RandomForestClasiifier works better on finding it is a rainy day
its better...
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv(r'C:\Users\ninjaac\Desktop\python\dataAnalysis\wether_RNN\weahterML\weatherHistory.csv\weatherHistory.csv')
#delete the null values
dataset.keys()
data=dataset.drop(columns=['Formatted Date','Summary','Daily Summary','Loud Cover',])

data=data.dropna(axis=0,how='any')
#no of output
data['Precip Type'].value_counts()
"""
rain    85224
snow    10712
Name: Precip Type, dtype: int64"""

# find the correation between the features
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
data['Precip Type']=la.fit_transform(data['Precip Type'])

#train test split
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.25,random_state=42)
#train count
train['Precip Type'].value_counts()
"""
rain    63914
snow     8038
Name: Precip Type, dtype: int64"""

test['Precip Type'].value_counts()
"""
rain    21310
snow     2674
Name: Precip Type, dtype: int64"""


#correlation

correlation_matrix=data.corr()
correlation_matrix['Precip Type'].sort_values(ascending=False)
"""
Precip Type                 1.000000
Humidity                    0.232622
Pressure (millibars)        0.009271
Wind Bearing (degrees)     -0.042142
Wind Speed (km/h)          -0.067771
Visibility (km)            -0.316483
Temperature (C)            -0.563503
Apparent Temperature (C)   -0.566058
Name: Precip Type, dtype: float64"""

#a strong negative correlation between the type andd theApparature tempareture 

#scatterd correlation ,atrix
from pandas.plotting import scatter_matrix
arguments=['Precip Type','Humidity','Visibility (km)','Temperature (C)']
scatter_matrix(data[arguments],figsize=(20,20))


#data creation
y_train=train.copy().pop('Precip Type')
X_train=train.drop(columns=['Precip Type'])

y_test=test.copy().pop('Precip Type')
X_test=test.drop(columns=['Precip Type'])

#data_op=data.copy().pop('Precip Type')
#data=data.drop(columns=['Precip Type'])
#keep data as dataframe or series not in arrays otherwise ti will gibe error in label encoder
#data=data.iloc[:,0].vlues return the numpy array so dont use this

#train test split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
#encoding the output

sca=StandardScaler()
X_train=sca.fit_transform(X_train)
sca_test=StandardScaler()
X_test=sca_test.fit_transform(X_test)

#random forest
from sklearn.ensemble import RandomForestClassifier
ran=RandomForestClassifier(n_estimators=30,n_jobs=-1,random_state=0)
ran.fit(X_train,y_train)
y_pred=ran.predict(X_test)

#accuracy score for randomForestClassifier
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
#Out[12]: 0.9637675116744496
accuracy_score(y_test, y_pred,normalize=False)
# 23115


#confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
"""
array([[21310,     0],
       [  869,  1805]], dtype=int64)"""

#svc model
from sklearn.svm import SVC

#grid search method for classification 
from sklearn.model_selection import GridSearchCV
parameters=[{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},{'kernel':['linear'],'C':[1,10,100,1000]}]

grid=GridSearchCV(SVC(),parameters,cv=5,n_jobs=-1)
#fit the model
grid.fit(X_train,y_train)
#feature details
print(f"the best parameters{grid.best_params_}")
#the best parameters{'C': 100, 'kernel': 'linear'}

print(f"the best scores{grid.best_score_}")
#the best scores0.9988603513453413

print(f"the grid search results{grid.cv_results_}")

#the complite report
from sklearn.metrics import classification_report
y_pred_grid=grid.predict(X_test)
classi_report=classification_report(y_test, y_pred_grid)
print(f"the classificaion report{classi_report}")
"""
the classificaion report              precision    recall  f1-score   support

           0       1.00      1.00      1.00     21310
           1       0.99      1.00      1.00      2674

    accuracy                           1.00     23984
   macro avg       1.00      1.00      1.00     23984
weighted avg       1.00      1.00      1.00     23984"""

#confusion matrix on SVC

from sklearn.metrics import confusion_matrix
confusion_matrix_SVC=confusion_matrix(y_test,y_pred_grid)
"""
array([[21286,    24],
       [    1,  2673]], dtype=int64)"""

#KNN model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)

#confusion matrix on KNN

from sklearn.metrics import confusion_matrix
confusion_matrix_KNN=confusion_matrix(y_test,y_pred_knn)
"""
Out[33]: 
array([[21204,   106],
       [  373,  2301]], dtype=int64)
"""

#accuracy score
accuracy_score(y_test, y_pred_knn)
#Out[34]: 0.9800283522348232







