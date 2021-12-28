# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:09:03 2020

@author: Kerim Demir
"""
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#import data
received_dataset = pd.read_csv("insurance/insurance.csv")
dataset = received_dataset.dropna()

#Plotting Data
plt.scatter(dataset.age,dataset.charges)
plt.xlabel("Age")
plt.ylabel("Charges")
plt.show()

#Linear Regression
linear_reg = LinearRegression()

x = dataset.age.values.reshape(-1,1)
y = dataset.charges.values.reshape(-1,1)

linear_reg.fit(x,y)

#Prediction
b0 = linear_reg.predict([[0]])
print("b0 =",b0)
b0_ = linear_reg.intercept_
print("b0_ =",b0_)
b1 = linear_reg.coef_
print("b1 =",b1)

print(linear_reg.predict([[43]]))

#Line Visualition

array = np.array(range(0,65)).reshape(-1,1)

y_head = linear_reg.predict(array)

plt.plot(array,y_head,color="red")
plt.show()










