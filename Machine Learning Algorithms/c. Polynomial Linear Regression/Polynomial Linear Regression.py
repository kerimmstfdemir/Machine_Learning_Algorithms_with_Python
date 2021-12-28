# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:19:31 2020

@author: Kerim Demir
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('polinomial_linear_regression_dataset.csv', sep = ';')

y = dataset.araba_max_hiz.values.reshape(-1,1)
x = dataset.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel('Max Hız (km/sa)')
plt.xlabel('Araba Fiyatı (x1000 TL)')
plt.show()

#%% Linear Regression

linear_reg = LinearRegression()
linear_reg.fit(x,y)

#%%
y_head = linear_reg.predict(x)

plt.plot(x,y_head, color = 'red', label = "linear")
plt.show()

#%%
# Polynomial Linear Regression : y = b0 + b1*x + b2*x^2 + b3*x^3 ...
from sklearn.preprocessing import PolynomialFeatures

polynomial_linear_regression = PolynomialFeatures(degree = 4)
x_polynomial = polynomial_linear_regression.fit_transform(x)

linear_reg2 = LinearRegression()
linear_reg2.fit(x_polynomial,y)

y_head2 = linear_reg2.predict(x_polynomial)

plt.plot(x,y_head2, color = "orange", label = "polynomial")
plt.legend()
plt.show()




