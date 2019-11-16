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



from sklearn.preprocessing import PolynomialFeatures

#===============================================================================
#create feed data for the model

#load csv file
sales = pd.read_csv('databases/kc_house_data.csv')
#print shape and name of columns
print('data frame shape: ', sales.shape)
print('data frame columns: \n', sales.columns)
#handeling the null values in the dataset
print('number of null values in each column:\n', sales.isnull().sum())
# if there are some null values you have two options:
#   1. fill the null fields with a space: modifiedSales = sales.fillna(“ “)
#   2. omit the the row with null fields: modifiedSales = sales.dropna()

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

#===============================================================================
#order=1 model train and evalute 

#create and fit linear regression model
model_o1 = LinearRegression()
model_o1.fit(X_train, y_train)

#aplly the model for predictting the test set outcome
test_set_predict = model_o1.predict(X_test)
#compute root mean squere error
rmse_o1 = np.sqrt(mean_squared_error(y_test,test_set_predict))
#compute r2 score, 1 is optimal
r2_o1 = r2_score(y_test,test_set_predict)
#print rsme and r2

print('model(order = 1) rmse: ', rmse_o1)
print('model r2(order = 1): ', r2_o1)


#===============================================================================
#order=2 model train and evalute

#compute higher order polynomial feed data
polynomial_features= PolynomialFeatures(degree=2)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.fit_transform(X_test)


#create and fit linear regression model
model_o2 = LinearRegression()
model_o2.fit(X_train_poly, y_train)


#aplly the model for predictting the test set outcome
test_set_predict = model_o2.predict(X_test_poly)
#compute root mean squere error
rmse_o2 = np.sqrt(mean_squared_error(y_test,test_set_predict))
#compute r2 score, 1 is optimal
r2_o2 = r2_score(y_test,test_set_predict)
#print rsme and r2

print('model(order = 2) rmse: ', rmse_o2)
print('model(order = 2) r2: ', r2_o2)

print('\n\nCode is Done\n')