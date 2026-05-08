import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from src.pipelines.prediction_pipeline import PredictionPipeline


app = FastAPI(title="Rossmann Sales Prediction API")

prediction_pipeline = PredictionPipeline()


class PredictionInput(BaseModel):
    store: int
    day_of_week: int
    promo: int
    state_holiday:str
    school_holiday: int
    date: datetime


@app.get("/")
def home():
    return {"message": "Rossmann Sales Prediction API Running"}


@app.post("/predict")
def predict(data: PredictionInput):

    input_df = pd.DataFrame([{
        "store": data.store,
        "dayofweek": data.day_of_week,
        "promo": data.promo,
        "stateholiday": data.state_holiday,
        "schoolholiday": data.school_holiday,
        "date": data.date
    }])


    prediction = prediction_pipeline.run_prediction_pipeline(df=input_df)
    print(f"prediction: {prediction}")

    return {
        "predicted_sales": float(prediction[0])
    }