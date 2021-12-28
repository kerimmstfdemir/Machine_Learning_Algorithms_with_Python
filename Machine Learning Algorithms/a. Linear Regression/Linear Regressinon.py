# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:50:00 2020

@author: Kerim Demir
"""

#import libraries
import pandas as pd
import matplotlib.pyplot as plt

#import data
df = pd.read_csv("linear_regression_dataset.csv", sep = ";")

#plotting data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("Deneyim (Yıl)")
plt.ylabel("Maaş")
plt.show()

#%% Linear Regression

#import sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

#%% Prediction
import numpy as np

b0 = linear_reg.predict([[0]])
print("b0 =",b0)

b0_ = linear_reg.intercept_
print("b0_ =",b0_)  #intercept point

b1 = linear_reg.coef_
print("b1 =",b1)    #slope

# maas = b0 + b1*deneyim

print(linear_reg.predict([[11]]))

#Line Visualization
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  #deneyim


y_head = linear_reg.predict(array)  #maas

plt.plot(array,y_head, color="red")
plt.show()




