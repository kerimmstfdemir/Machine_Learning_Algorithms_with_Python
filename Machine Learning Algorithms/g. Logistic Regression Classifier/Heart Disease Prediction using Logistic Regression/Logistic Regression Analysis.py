# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:51:11 2020

@author: Kerim Demir
"""


#%%import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%read csv file
dirty_dataset = pd.read_csv("heart-disease-prediction-using-logistic-regression/framingham.csv")
dirty_dataset.isna()
dataset = dirty_dataset.dropna(axis = 0)
print(dataset.info())

x_data = dataset.drop(["TenYearCHD"],axis = 1)
y = dataset.TenYearCHD.values

#%% Normalization
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)))

#%% Test - Train Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 42)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train :",x_train.shape)
print("x_test  :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test  :",y_test.shape)

#%% sklearn with Linear Regression
from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression()
logistic_reg.fit(x_train.T,y_train.T)

print("Test Accuracy : {}".format(logistic_reg.score(x_test.T,y_test.T)))