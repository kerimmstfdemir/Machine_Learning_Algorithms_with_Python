# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:22:01 2020

@author: Kerim Demir
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

#%%
# Basic Linear Regression (pelvic_incidence - Class)
"""
from sklearn.linear_model import LinearRegression

dataset["class"].replace("Abnormal",0, inplace = True)
dataset["class"].replace("Normal",1, inplace = True)

x = dataset.pelvic_incidence.values.reshape(-1,1)
y = dataset.iloc[:,6].values.reshape(-1,1)

linear_reg = LinearRegression()
linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

plt.scatter(x,y)
plt.xlabel("pelvic_incidence")
plt.ylabel("class")
plt.plot(x,y_head, color = 'red')
plt.show()

from sklearn.metrics import r2_score

print("r2 score :",r2_score(y,y_head))  # r2 score = 0.12484622036637794
"""
#%%
# Multiple Linear Regression (All variables - Class)
"""
from sklearn.linear_model import LinearRegression

dataset["class"].replace("Abnormal",0, inplace = True)
dataset["class"].replace("Normal",1, inplace = True)

x = dataset.iloc[:,[0,1,2,3,4,5]].values
y = dataset.iloc[:,6].values.reshape(-1,1)

linear_reg = LinearRegression()
linear_reg.fit(x,y)

y_head = linear_reg.predict(x)

print(linear_reg.predict([[53.43,15.86,37.16,37.56,120.56,5.98]]))

from sklearn.metrics import r2_score

print("r2 score :",r2_score(y,y_head)) # r2 score = 0.34131068482390325

"""
#%%
# Polynomial Linear Regression
"""
from sklearn.linear_model import  LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset["class"].replace("Abnormal",0, inplace = True)
dataset["class"].replace("Normal",1, inplace = True)

x = dataset.iloc[:,[0,1,2,3,4,5]].values
y = dataset.iloc[:,6].values.reshape(-1,1)

Polynomial_Linear_Reg = PolynomialFeatures(degree = 7)
x_polynomial = Polynomial_Linear_Reg.fit_transform(x)

linear_reg = LinearRegression()
linear_reg.fit(x_polynomial,y)  

y_head = linear_reg.predict(x_polynomial)

from sklearn.metrics import r2_score

print("r2 score :",r2_score(y,y_head)) # r2 score = 0.9999998959683727 (for degree = 7)
"""
#%%
#Decision Tree Regression
"""
from sklearn.tree import DecisionTreeRegressor

dataset["class"].replace("Abnormal",0, inplace =True)
dataset["class"].replace("Normal",1, inplace = True)

x = dataset.iloc[:,[0,1,2,3,4,5]].values
y = dataset.iloc[:,6].values.reshape(-1,1)

decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(x,y)

y_head = decision_tree_reg.predict(x)

print(decision_tree_reg.predict([[63,22,39,40.52,150,-0.56]]))

from sklearn.metrics import r2_score

print("r2 score :",r2_score(y,y_head))  # r2 score = 1.0
"""
#%%
#Rando Forest Regression
from sklearn.ensemble import RandomForestRegressor

dataset["class"].replace("Abnormal",0, inplace =True)
dataset["class"].replace("Normal",1, inplace = True)

x = dataset.iloc[:,[0,1,2,3,4,5]].values
y = dataset.iloc[:,6].values.reshape(-1,1)

random_forest_reg = RandomForestRegressor(n_estimators = 100, random_state = 42)
random_forest_reg.fit(x,y)

y_head = random_forest_reg.predict(x).reshape(-1,1)

from sklearn.metrics import r2_score

print("r2 score :",r2_score(y,y_head))  # r2 score = 0.9260295714285715

#%%