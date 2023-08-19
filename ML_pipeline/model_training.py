#import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble  import VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def split_train_test(df, target_variable, size, seed):
    '''df:dataframe
       target_variable:target feature name
       size: test size ratio
       seed: random state'''
    try:
        x=df.drop(target_variable, axist=1)
        y = df[[target_variable]]
        x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=size,random_state=seed)
    except Exception as e: 
        print(e)
    else:
        return x_train, x_test, y_train, y_test    
    

def regression_model_training(x_train, x_test, y_train, y_test):
    '''
    Script to train linear reqression and regularization models
    :param x_train: training split
    :param y_train: training target vector
    :param x_test: test split
    :param y_test: test target vector 
    :return: Dataframe of model evaluation, model objects'''
    try:
        models=[
            ('linear_Reg', LinearRegression()),
            ('ridge', Ridge()),
            ('lasso', Lasso()),
         ]
        output= pd.DataFrame()
        comparison_column= ['Model','Train_R2','Train-MAE','Train_RMSE','Test_R2','Test_MAE','Test_RMSE']
        estimators= []
        for name, model in models:

            rgr =model.fit(x_train, y_train)
            y_pred_train =rgr.predict(x_train)
            y_pred_test= rgr.predict(y_train)

            #MAIN ABSOLUTE ERROR OR MME
            MAE_train = round(mean_absolute_error(y_train,y_pred_train),6 )
            MAE_test = round(mean_absolute_error(y_test,y_pred_test), 6)

            #ROOT MEAN SQUARED ERROR OR RMSE
            RMSE_train= round(mean_squared_error(y_train, y_pred_train, squared=False), 6)
            RMSE_test= round(mean_absolute_error(y_test, y_pred_test, squared=False), 6)

            #R2
            R2_train= round(r2_score(y_train, y_pred_train), 6)
            R2_test= round(r2_score ( y_test, y_pred_test), 6)
            estimators.append(rgr)
            
            metric_score= [name, R2_train,MAE_train, RMSE_train, R2_test, MAE_test, RMSE_test]
            final_dict= dict(zip(comparison_column,metric_score))
            df_dictionary= pd.DataFrame((final_dict))
            output= pd.concat([output, df_dictionary], ignore_index=True)

    except Exception as e:
        print(e)
    else:
        return output, estimators[0], estimators[1], estimators[2]
    







def ensemble_regression(x_train, x_test, y_train, y_test, estimators):
    '''
    script to train a voting regressor
    estimator:list of tuples of name and fitted regressor objects'''
    try:
        comparison_columns= ['Model','Train_R2','Train-MAE','Train_RMSE','Test_R2','Test_MAE','Test_RMSE']
        #TRAIN
        votiing_ensemble= VotingRegressor(estimators,)
        votiing_ensemble.fit(x_train, y_train)

        #PREDICT
        y_pred_train = votiing_ensemble.predict(x_train)
        y_pred_test = votiing_ensemble.predict(x_test)

        #MEAN ABSOLUTE ERROR
        MAE_train = round(mean_absolute_error (y_train,y_pred_train),6 )
        MAE_test = round(mean_absolute_error(y_test,y_pred_test), 6)

        #ROOT MEAN SQUARED ERROR OR RMSE
        RMSE_train= round(mean_squared_error(y_train, y_pred_train, squared=False), 6)
        RMSE_test= round(mean_absolute_error(y_test, y_pred_test, squared=False), 6)

        #R2
        R2_train= round(r2_score(y_train, y_pred_train), 6)
        R2_test= round(r2_score ( y_test, y_pred_test), 6)
        
        #COMPARISON DATAFRAME
        metric_scores=['Voting_Ensemble',R2_train, MAE_train, RMSE_train,R2_test, MAE_test, RMSE_test,]
        final_dict=dict(zip(comparison_columns, metric_scores))
        df_dictionary=pd.DataFrame([final_dict])
    

    except Exception as e:
        print(e)
    else:
        return df_dictionary, votiing_ensemble


    
    


            