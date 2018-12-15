#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:17:39 2018

@author: Nihat Allahverdiyev
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

iris_dataset = load_iris()

X_train,  X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                        marker='o',hist_kwds={'bins': 20}, s=60,
                        alpha=.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset
      ['target_names'][prediction]))

y_predict = knn.predict(X_test)

print("Predicted: {}".format(y_predict))

print("Test score: {:.2f}".format(np.mean(y_predict == y_test)))
