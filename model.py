# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:42:56 2018

@author: Utkarsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:len(dataset.iloc[0,]) - 1].values
y = dataset.iloc[:, len(dataset.iloc[0,])-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 42)
regressor.fit(x,y)

#print(regressor.predict(605))

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x,y, color = 'red', marker = '*')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.show()