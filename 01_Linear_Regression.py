'''
In this file an instance of Linear regression class
will be used to be trained by boston house dataset.
'''

import pandas as pd
import numpy as np
import ML_Lib as ml
from sklearn import preprocessing
# dataset
from sklearn.datasets import load_boston

#Y is price of the homes
#X is the input data
X, Y = load_boston(return_X_y=True)
print('type of data: ', type(X))
print('shape of X: ', X.shape)
print('shape of Y: ', Y.shape)
#reshape Y
Y = Y[:, np.newaxis]
print('shape of Y after reshaping:', Y.shape)

#create an instance of ridge regression class
a = ml.ridge_regression_class(X, Y, name_of_saved_model='ridge01.sav', max_alpha=200)
a.run()
#create an instance of ridge regression perdict class
model_input = preprocessing.scale(X[0,:])
model_input = model_input.reshape([1, 13])
b = ml.ridge_regression_perdict_class('ridge01.sav', model_input)
b.mpredict()