''' 
In this file linear regression will be considered to do predictions
Also, an instance of Linear regression class will be used to be trained by the dataset.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

#==================================================================
#create feed data for the model

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
#create feature dataset considering all columns except... 
#...date for being not integer and obviously the price
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

#splitting the sales_features and target to train and test sets
X_train, X_test, y_train, y_test = train_test_split(sales_features, 
                                                    target,
                                                    test_size=.3,
                                                    random_state = 0)

#==================================================================
#model train and evalute

#create and fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#aplly the model for predictting the test set outcome
test_set_predict = model.predict(X_test)
#compute root mean squere error
rmse = np.sqrt(mean_squared_error(y_test,test_set_predict))
#compute r2 score, 1 is optimal
r2 = r2_score(y_test,test_set_predict)
#print rsme and r2

print('model rmse: ', rmse)
print('model r2: ', r2)




print('\n\nCode is Done\n')