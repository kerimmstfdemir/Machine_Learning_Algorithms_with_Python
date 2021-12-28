# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:57:50 2020

@author: Kerim Demir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("original_dataset.csv", sep = ";")

x = data.iloc[:,[0,2]]
y = data.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

print("b0 =",multiple_linear_regression.intercept_)
print("\nb1,b2 =",multiple_linear_regression.coef_)

#Prediction
multiple_linear_regression.predict([[10,35],[5,35]])

#3D Plotting in Multiple Linear Regression

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data.iloc[:,[0]]
y = data.iloc[:,[2]]
z = data.maas.values.reshape(-1,1)

array1 = np.array(range(0,16)).reshape(-1,1)
array2 = np.array(range(21,37)).reshape(-1,1)
value_head = []

for i in array1:
    for j in array2:  
        head = multiple_linear_regression.predict([[int(i),int(j)]])
        value_head.append(head)

value_head = np.array(value_head).reshape(-1,1) #value_head reshaped

ax.scatter(x,y,z, color = 'red', marker = 'o', alpha = 0.5)
ax.plot_surface(array1,array2,value_head.reshape(16,-1),color = 'None',alpha = 0.3)
ax.set_xlabel('Deneyim')
ax.set_ylabel('Yaş')
ax.set_zlabel('Maaş')


