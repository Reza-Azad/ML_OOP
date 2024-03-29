'''
ridge regression model is generally better than 
the OLS model in prediction
In this code Ridge Regression class will be 
used to train boston house data set
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

#create an instance of ridge regression class
Boston_RR = ml.ridge_regression_class(X, Y, name_of_saved_model='ridge01.sav', max_alpha=200)
Boston_RR.run()

#first row of data set for testing the trained model
#for Ridge Regression the input for the model should be scaled
model_input = preprocessing.scale(X[0,:]) 
model_input = model_input.reshape([1, 13])

#create an instance of ridge regression predict class
Boston_predict = ml.ridge_regression_perdict_class('ridge01.sav', model_input)
Boston_predict.mpredict()


print('\n\ncode is done')