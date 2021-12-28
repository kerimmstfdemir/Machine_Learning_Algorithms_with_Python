# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:36:34 2020

@author: Kerim Demir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('decision_tree_regression_dataset.csv', sep =';', header = None)

x = dataset.iloc[:,0].values.reshape(-1,1)
y = dataset.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.tree import DecisionTreeRegressor

decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(x,y)

print(decision_tree_reg.predict([[5.5]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = decision_tree_reg.predict(x_)

#%% Visulation Part
plt.scatter(x,y,color = 'red')
plt.plot(x_,y_head,color = 'green')
plt.xlabel('Tribun Level')
plt.ylabel('Price')
plt.show()