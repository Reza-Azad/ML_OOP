''' 
In this file linear regression will be considered to do predictions
also, an instance of Linear regression class will be used to be trained by the dataset.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

#load csv file
sales = pd.read_csv('databases/kc_house_data.csv')
#print shape and name of columns
print('data frame shape: ', sales.shape)
print('data frame columns: \n', sales.columns)
#handeling the null values in the dataset
print('print the number of null values in each column', sales.isnull().sum())
# if there are some null values you have two options:
#   1. fill the null fields with a space: modifiedSales = sales.fillna(“ “)
#   2. omit the the row with null fields: modifiedSales = sales.dropna()

print(sales.head())

#create a copy to protect the original dataset from manipulation
sales_copy = sales.copy()
#create feature dataset
sales_features_dataset = sales_copy.drop(columns=['price', 'date'])
#create feature nparray (X)
sales_features = np.array(sales_features_dataset)
#create target nparray (Y)
target = np.array(sales['price'])

print('shape of sales_feature: ', sales_features.shape)
print('shape of target:', target.shape)
#reshape target array to 2D
target = target[:, np.newaxis]
print('shape of target after reshape to 2D:', target.shape)



X_train, X_test, y_train, y_test = train_test_split(sales_features, 
                                                    target,
                                                    test_size=.3,
                                                    random_state = 0)

print('shape of X_train:', X_train.shape)
print('shape of Y_train:', y_train.shape)
#create and fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)





print('\n\nCode is Done\n')