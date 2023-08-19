import pandas as pd
import numpy as np
import pickle
import os
from scipy import stats
import sklearn



#function to read the data
def read_data(file_path, **kwargs):
    try:
        raw_data = pd.read_excel(file_path, **kwargs)
    except Exception as e:
        print(e)
    else:
        return raw_data
    



#function to dump python objects
def pickle_dump(data, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(e)




#function for prediction interval
def get_prediction_interval(prediction, actual_values, predicted_values, pi=.95):
    """
    get a prediction interval for the regression model.
    
    INPUTS:
           - Single prediction (test data)
           - y_train
           - prediction from x_train
           -  prediction interval thresold (default = .95)\
            
    OUTPUT
           - prediction interval for single test prediction 

           """
    try:
        #get standard deviation of prediction on the train dataset
        sum_errs= np.sum((actual_values - predicted_values)**2) 
        stdev = np.sqrt(sum_errs/ (len(actual_values)-1))


        #get interval from standard deviation
        one_minus_pi = 1 - pi
        ppf_lookup = 1 -(one_minus_pi/ 2) # if we need to calculate a Two-tail test" (i.e we are concerned with values both greater and less than our mean then we need to split the significant (i.e our alpha value) because  we are still using a calculation method for one-tail. the split in half symbolizesthe significance level being appropriated to both tails. a 95% significance level has a 5% alpha: splitting the 5% alpha across both tails return 2.5%. taking 2.5% from 100% returns 97.5% as input for the significance level)
        z_score = stats.norm.ppf(ppf_lookup) # this will return a valu (that function as a standard-deviation multiplier) making where 95% (pi%) of data points would be contained if our data is a normal distribution.



        #get interval from standard deviation
        one_minus_pi = 1 -pi
        ppf_lookup = 1 -(one_minus_pi/ 2)
        z_score = stats.norm.ppf(ppf_lookup)
        interval = z_score * stdev
        


        #generate prediction interval lower and upper bound cs_24
        lower, upper = prediction-interval, prediction + interval
        return lower[0], upper[0]
    except Exception as e:
        print(e)




def price_range(model_obj, x_train, x_test, y_train):
    """the function takes in fitted model objects and returns a data frame for lower and upper price ranges
    using prediction or confidence interval"""

    #getting prediction intervals for the test data
    try:
        lower_vet=[]
        upper_vet=[]
        if isinstance(model_obj, sklearn.linear_model.Lasso)==True or isinstance(model_obj, sklearn.ensemble.VotingRegressor):
            preds_test= model_obj.predict(x_test).reshape(-1,1)
        else:
            preds_test= model_obj.predict(x_test)
        for i in preds_test:
            lower, upper = get_prediction_interval(i, y_train.value, model_obj.predict(x_train).reshape(-1, 1))
            lower_vet.append(lower)
            upper_vet.append(upper)

        df= pd.DataFrame(zip(lower_vet, upper_vet, preds_test.reshape(-1).tolist()), columns=["lower","upper","mean"])
        return df
    except Exception as e:
        print(e)


