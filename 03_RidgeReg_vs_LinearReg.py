'''
In this file an instance of Linear regression class
will be used to be trained by boston house dataset.
'''

import pandas as pd
import numpy as np
import ML_Lib as ml
from sklearn import preprocessing
from sklearn.datasets import load_boston # dataset

#===============================================================================
#prepare feed data

#Y is price of the homes
#X is the input data
X, Y = load_boston(return_X_y=True)
print('shape of X: ', X.shape)
print('shape of Y: ', Y.shape)
#reshape Y
Y = Y[:, np.newaxis]
print('shape of Y after reshaping:', Y.shape)


#===============================================================================
#linear regression section

#create an instance of linear regression calss
Boston_LR = ml.linear_regression_class(X, Y, name_of_saved_model='linear_regression.sav')
Boston_LR.run()

#first row of dataset for testing the trained (linear regression) model
model_input = X[0,:]
model_input = model_input.reshape([1, 13])
#create an instance of linear regression Predict class for loading the saved model
print('\nResult of Linear rigression:')
Boston_Predict_LR = ml.linear_regression_predict_class('linear_regression.sav', model_input)
Boston_Predict_LR.mpredict()

#===============================================================================
#ridge regression section

#create an instance of ridge regressoin class
Boston_RR = ml.ridge_regression_class(X, Y, name_of_saved_model='ridge_r.sav')
Boston_RR.run()

#first row of data set for testing the trained model
#for Ridge Regression the input for the model should be scaled
model_input = preprocessing.scale(X[0,:]) 
model_input = model_input.reshape([1, 13])

#create an instance of ridge regression predict class
print('\nResult of Ridge rigression:')
Boston_predict_RR = ml.ridge_regression_perdict_class('ridge_r.sav', model_input)
Boston_predict_RR.mpredict()

#display first element of Y to compare Ridge and linear regression
print('\nfirst element of Y to compare the result of Ridge and linear regression: ', Y[0])