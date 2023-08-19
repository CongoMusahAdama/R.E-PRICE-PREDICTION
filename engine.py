#importing libraries
import pandas as pd
from ML_pipeline import utils
from ML_pipeline import preprocessing
from ML_pipeline import model_training
import os
import warnings
warnings.filterwarnings('ignore')


#select the script to run, training script1 or web application script 2
script_number= int(input("Enter 1 to train the model\n Enter 2 to use te fastapi aplication:"))

#model trainign script
if script_number==1:

    #reading the data
    df= utils.read_data('input\Pune real estate Data.xlsx')

    #preprocess data and generates features
    df1= preprocessing.preprocess_data(df)


# taining the model
#train test slit
    x_train, x_test, y_train, y_test = model_training.split_train_test(df1,'Price_in_Takoradi',0.3,1234)
#training regressors
    output,linear_reg, ridge, las=model_training.regression_model_training(x_train,x_test,y_train,y_test)

#training ensemble
    estimators =[('lr',linear_reg),('rid',ridge),('lasso',las)]
    df_dict, voting_ensemble=model_training.ensemble_regression(x_train,x_test,y_train, y_test, estimators)
    output= pd.concat([output, df_dict], ignore_index=True)
    

#training regressors
    estimators =[('lr',linear_reg),('rid',ridge),('lasso',las)]
    df_dict, voting_ensemble=model_training.ensemble_regression(x_train,x_test,y_train, y_test, estimators)
    output= pd.concat([output, df_dict], ignore_index=True)
    output.head()



#price range prediction
    dflinear=utils.price_range(linear_reg, x_train,x_test,y_train)
    dfridge=utils.price_range(ridge, x_train,x_test,y_train)
    dflasso=utils.price_range(las,x_train,x_test,y_train)
    dfensemble= utils.price_range(voting_ensemble, x_train,x_test,y_train)

#saving model
    utils.pickle_dump(voting_ensemble,'output/property_price_prediction_voting.sav')

#end
    print("Voting Regressor model has been trained and saved in output folder!")

elif script_number==2:
    #terminal command to un the fastapi
    os.system("uvicorn app:PropertyPricePredApp --reload")

else:
     print("enter a valid script number")
    