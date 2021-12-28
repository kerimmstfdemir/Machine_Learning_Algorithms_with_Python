# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:40:25 2020

@author: Kerim Demir
"""


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
dataset = pd.read_csv("biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

A = dataset[dataset["class"] == "Abnormal"]
N = dataset[dataset["class"] == "Normal"]

#scatter plot
plt.scatter(A.pelvic_incidence,A.degree_spondylolisthesis, color = "red", alpha = 0.3)
plt.scatter(N.pelvic_incidence,N.degree_spondylolisthesis, color = "green", alpha = 0.3)
plt.xlabel("pelvic_incidence")
plt.ylabel("degree_spondylolisthesis")
plt.legend()
plt.show()

dataset["class"] = [1 if each == "Normal" else 0 for each in dataset["class"]]
y = dataset["class"].values
x_data = dataset.drop(["class"],axis = 1)

#Normalization
x = ((x_data - np.min(x_data)) / ((np.max(x_data)) - np.min(x_data)))

#Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

#kNN Model
from sklearn.neighbors import KNeighborsClassifier
kNN = KNeighborsClassifier(n_neighbors=22)  #n_neighbors = k value
kNN.fit(x_train,y_train)
prediction = kNN.predict(x_test)

print("kNN Score (k = {}) : {}".format(22,kNN.score(x_test,y_test)))

#Finding best k value
score_list = []
for each in range(1,150):
    kNN2 = KNeighborsClassifier(n_neighbors=each)
    kNN2.fit(x_train,y_train)
    score_list.append(kNN2.score(x_test,y_test))
    
plt.plot(range(1,150),score_list)
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.show()