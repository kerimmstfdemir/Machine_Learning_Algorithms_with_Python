# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 08:32:28 2020

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
y_head = linear_reg.predict(x)  #maas

plt.plot(x,y_head, color="red")
plt.show()

#%%
from sklearn.metrics import r2_score

print("r2 score :",r2_score(y,y_head))




