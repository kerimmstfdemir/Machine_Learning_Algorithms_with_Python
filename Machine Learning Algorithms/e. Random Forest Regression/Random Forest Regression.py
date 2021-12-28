# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:11:59 2020

@author: Kerim Demir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('random_forest_regression_dataset.csv',sep = ';',header = None)

x = dataset.iloc[:,0].values.reshape(-1,1)
y = dataset.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor

random_forest_reg = RandomForestRegressor(n_estimators = 100, random_state = 42)
random_forest_reg.fit(x,y)

print("7.8 için fiyat tahmini :",random_forest_reg.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = random_forest_reg.predict(x_).reshape(-1,1)

plt.scatter(x,y,color = "red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("Tribun Level")
plt.ylabel("Price")
plt.show()