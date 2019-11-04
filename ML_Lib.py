from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sklearn import preprocessing


class linear_regression_class():
    '''class for linear regression | 
       initialize class and then execute run()'''
    
    def __init__(self, X, Y, test_trian_ratio=.2, name_of_saved_model='finalized_model.sav'):
        #check for X, Y dimension match
        #assert X.shape == Y.shape , 'X, Y dimension MISMATCH'
        self.X = X
        self.Y = Y 
        self.test_trian_ratio = test_trian_ratio
        self.name_of_saved_model = name_of_saved_model
        
    def __mspilit_data__(self):
        '''split input X, Y in to train and test groups
           returns X_train, X_test, y_train, y_test'''
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.Y,
                                                            test_size=self.test_trian_ratio,
                                                            random_state = 0)
        return X_train, X_test, y_train, y_test
        
    
    def __model_train__(self,X_train, X_test, y_train, y_test):
        '''train linear regression model |
           returns model, rmse, r2 '''
        
        #create and fit linear regression model
        model = LinearRegression(normalize=True)
        model.fit(X_train, y_train)

        #compute prediction for X_test
        y_pred_test = model.predict(X_test)

        #compute root mean squere error
        rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
        #compute r2 score, 1 is optimal
        r2 = r2_score(y_test,y_pred_test)
        
        print('min RMSE: ', rmse)
        print('corresponding r2: ', r2)




    def __train_final_model__(self):
        '''train linear regression with whole data (without
        splitting) '''
        
        print('final model training is in progress...')
        
        #create and fit linear regression model
        final_model = LinearRegression()
        final_model.fit(self.X, self.Y)

        # save the model to disk
        pickle.dump(final_model, open(self.name_of_saved_model, 'wb'))

        print('final model is saved to directory and ready to do predictions')
        

    def __print_result__(self,min_rmse, corres_r2, best_poly_order, best_model ):
        print('ratio of test to train dataset:  ', self.test_trian_ratio)
        print('best plynomial order: ', best_poly_order)
        print('min RMSE: ', min_rmse)
        print('corrseponding r2: ', corres_r2)
        #print('coef of the model:', best_model.coef_) #uncomment this line to see the coefficients of the model
        #print('rank of the model: ', best_model.rank_) uncomment this line to see the rank of the model
    
    def run(self):
        '''run model trianing process '''
        X_train, X_test, y_train, y_test = self.__mspilit_data__()
        self.__model_train__(X_train, X_test, y_train, y_test)
        #self.__print_result__(min_rmse, corres_r2, best_poly_order, best_model)
        self.__train_final_model__()      

 
class linear_regression_predict_class():
    '''class for loading a saved linear regression model and
       compute predictions | after initialization execute mpredict() '''
    def __init__(self, str_model_full_name, model_input):
        self.model_name = str_model_full_name
        self.model_input = model_input
    
    def mpredict(self):
        '''compute prediction of input '''
        # load the model from disk
        loaded_model = pickle.load(open(self.model_name, 'rb'))
        if loaded_model:
            print('\nmodel loaded successfully')
        else:
            print('\nFailed to load the model')
               
        print('predicted value for ', self.model_input, ': ', loaded_model.predict(self.model_input))

# usful links for linear regression
# https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
# https://datatofish.com/multiple-linear-regression-python/
# https://towardsdatascience.com/polynomial-regression-bbe8b9d97491
# https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86

class ridge_regression_class():
    '''class for ridge regression'''
    
    def __init__(self, X, Y, test_trian_ratio=.2, max_alpha=200, name_of_saved_model='finalized_model.sav'):
        #check for X, Y dimension match
        #assert X.shape == Y.shape , 'X, Y dimension MISMATCH'
        self.X = X
        self.Y = Y 
        self.test_trian_ratio = test_trian_ratio
        #max alpha for ridge regression
        self.max_alpha = max_alpha + 1 #since python exludes last member of range
        self.name_of_saved_model = name_of_saved_model
        
    def __mspilit_scale_data__(self):
        '''split input X, Y in to train and test groups
           and scale(standardize) x | 
           returns X_train, X_test, y_train, y_test'''
        
        #scale self.X
        self.X = preprocessing.scale(self.X)
        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.Y,
                                                            test_size=self.test_trian_ratio,
                                                            random_state = 0)
        return X_train, X_test, y_train, y_test
        
    
    def __model_train__(self,X_train, X_test, y_train, y_test, malpha):
        '''train ridge regression model |
           returns model, rmse, r2 '''
        
        #create and fit ridge regression model
        model = Ridge(alpha=malpha)
        model.fit(X_train, y_train)

        #compute prediction for X_test
        y_pred_test = model.predict(X_test)

        #compute root mean squere error
        rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))
        #compute r2 score, 1 is optimal
        r2 = r2_score(y_test,y_pred_test)
        
        return model, rmse, r2

    def __find_best_model__(self,X_train, X_test, y_train, y_test):
        '''find best alpha for model by minimum rmse (root mean square error)
        | returns min_rmse, corresponding_r2, best_alpha  '''
        
        print('Training is in progress ...')        
        for i in range(1, self.max_alpha): 
            model, rmse, r2 = self.__model_train__(X_train, X_test, y_train, y_test, i)
            if i == 1 :
                print('best alpha is:', i)
                min_rmse = rmse
            if rmse <= min_rmse :
                min_rmse = rmse
                corresponding_r2 = r2
                best_alpha = i
                best_model = model
    
        return min_rmse, corresponding_r2, best_alpha, best_model

    def __train_final_model__(self, best_alpha):
        '''trian ridge regression with whole data (without splitting) '''
        
        print('final model training is in progress...')
        
        #creat and fit ridge regression model
        final_model = Ridge(alpha=best_alpha)
        final_model.fit(self.X, self.Y)

        # save the model to disk
        pickle.dump(final_model, open(self.name_of_saved_model, 'wb'))

        print('final model is saved to directory and ready to do predictions')
        

    def __print_result__(self,min_rmse, corres_r2, best_alpha, best_model ):
        print('ratio of test to train dataset:  ', self.test_trian_ratio)
        print('best alpha: ', best_alpha)
        print('min RMSE: ', min_rmse)
        print('corrseponding r2: ', corres_r2)
        #print('coef of the model:', best_model.coef_) #uncomment this line to see the coefficients of the model
        #print('rank of the model: ', best_model.rank_) uncomment this line to see the rank of the model
    
    def run(self):
        '''run model trianing process '''
        X_train, X_test, y_train, y_test = self.__mspilit_scale_data__()
        min_rmse, corres_r2, best_alpha, best_model =self.__find_best_model__(X_train, X_test, y_train, y_test)
        self.__print_result__(min_rmse, corres_r2, best_alpha, best_model)
        self.__train_final_model__(best_alpha)      
    
class ridge_regression_perdict_class():
    '''class for loading a saved ridge regression model and
       compute predictions | after initialization execute mpredict()'''
    def __init__(self, str_model_full_name, model_input):
        self.model_name = str_model_full_name
        self.model_input = model_input
    
    def mpredict(self):
        '''compute prediction of input '''
        # load the model from disk
        loaded_model = pickle.load(open(self.model_name, 'rb'))
        if loaded_model:
            print('\nmodel loaded successfully')
        else:
            print('\nFailed to load the model')
        
        print('predicted value for ', self.model_input, ': ', loaded_model.predict(self.model_input))
