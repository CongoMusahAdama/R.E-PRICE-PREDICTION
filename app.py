#import uvicorn
from fastapi import FastAPI
from PropertyVariables import PropertyPricePred
import pandas as pd
import joblib

# creating the App object
PropertyPricePredApp= FastAPI()

# load the model from disk

fileName='PropertyPricePredApp'
loaded_model= joblib.load(fileName)


#index route, opens automatically on http:// 127.0.0.1:8000
@PropertyPricePredApp.get("/")
def indef():
    return {"mesage":"hello, world i got something awesome for you for your real estate price pred"}



# expose the prediction  functionality, make a prediction from the passed.
# JSON data and return the predicted price with the confidence (http:///127.0.0.1.8000/predict)
@PropertyPricePredApp.post("/predict")
def predict_price(data: PropertyPricePred):
    data= data.dict()
    print(data)
    data=pd.Dataframe([data])
    print(data.head())

    prediction = loaded_model.predict(data)
    print(str(prediction))
    return str(prediction)


# run the API at the terminal with uvicorn
#uvicorn app: PropertyPricePredApp --reload