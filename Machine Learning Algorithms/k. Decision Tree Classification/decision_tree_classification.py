# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:07:48 2020

@author: Kerim Demir
"""


#importing libraries
import numpy as np
import pandas as pd

#%% reading dataset
dataset = pd.read_csv("decision_tree_classification_dataset.csv")

dataset.drop(["id","Unnamed: 32"],axis =1, inplace = True)
dataset.diagnosis = [1 if each == "M" else 0 for each in dataset.diagnosis]

y = dataset.diagnosis.values
x_data = dataset.drop(["diagnosis"],axis = 1)

#%% Normalization
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)))

#%% Train - Test Split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.15, random_state = 42)

#%%
from sklearn.tree import DecisionTreeClassifier

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(x_train,y_train)

print("Decision Tree Classification Accuracy :",decision_tree_classifier.score(x_test,y_test))

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

y_pred = decision_tree_classifier.predict(x_test)
print("r2 Score :",r2_score(y_test,y_pred))
print("Accuracy Score :",accuracy_score(y_test,y_pred))


