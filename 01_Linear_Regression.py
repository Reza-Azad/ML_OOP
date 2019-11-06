'''
In this file an instance of Linear regression class
will be used to be trained by boston house dataset.
'''

import pandas as pd
import numpy as np
import ML_Lib as ml
from sklearn import preprocessing
from sklearn.datasets import load_boston # dataset

#Y is price of the homes
#X is the input data
X, Y = load_boston(return_X_y=True)
print('shape of X: ', X.shape)
print('shape of Y: ', Y.shape)
#reshape Y
Y = Y[:, np.newaxis]
print('shape of Y after reshaping:', Y.shape)

#create an instance of linear regression calss
Boston_LR = ml.linear_regression_class(X, Y, name_of_saved_model='linear_regression.sav')
Boston_LR.run()

#first row of dataset for testing the trained model
model_input = X[0,:]
model_input = model_input.reshape([1, 13])
#create an instance of linear regression Predict class for loading the saved model
Boston_Predict = ml.linear_regression_predict_class('linear_regression.sav', model_input)
Boston_Predict.mpredict()

