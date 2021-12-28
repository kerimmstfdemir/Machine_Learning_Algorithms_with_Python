# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:37:02 2020

@author: Kerim Demir
"""


#%%importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read csv
dataset = pd.read_csv("naive_bias_dataset.csv")

#%%
dataset.drop(["id","Unnamed: 32"],axis = 1, inplace = True)
# Malignant = 'M'
# Benign    = 'B'

#%%
M = dataset[dataset.diagnosis == "M"]
B = dataset[dataset.diagnosis == "B"]

# Scatter Plot
plt.scatter(M.radius_mean,M.texture_mean,color = 'red', label = "Malignant", alpha = 0.3)
plt.scatter(B.radius_mean,B.texture_mean,color = 'green', label = "Benign", alpha = 0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

#%%
dataset.diagnosis = [1 if each == "M" else 0 for each in dataset.diagnosis]
y = dataset.diagnosis.values
x_data = dataset.drop(["diagnosis"],axis = 1)

#%%
#Normalization
x = ((x_data - np.min(x_data)) / ((np.max(x_data)) - np.min(x_data)))

#%%
#Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state =1)

#%% Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
naive_bayes.fit(x_train,y_train)

#%% Accuracy
print("Accuracy of Naive Bayes Algorithm :",naive_bayes.score(x_test,y_test))
