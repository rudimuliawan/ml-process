import json

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocessing import preprocess_data
from src.utils import deserialize_data


class Item(BaseModel):
    age: float
    income: float
    home_ownership: str
    employment_length: float
    loan_intent: str
    loan_grade: str
    loan_amount: float
    loan_interest_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    credit_history_length: float

    def to_dataframe(self):
        return pd.DataFrame([self.model_dump()])


class CleanedItem(BaseModel):
    age: float
    income: float
    employment_length: float
    loan_amount: float
    loan_interest_rate: float
    loan_percent_income: float
    credit_history_length: float
    home_ownership_MORTGAGE: float
    home_ownership_OTHER: float
    home_ownership_OWN: float
    home_ownership_RENT: float
    loan_intent_DEBTCONSOLIDATION: float
    loan_intent_EDUCATION: float
    loan_intent_HOMEIMPROVEMENT: float
    loan_intent_MEDICAL: float
    loan_intent_PERSONAL: float
    loan_intent_VENTURE: float
    loan_grade_A: float
    loan_grade_B: float
    loan_grade_C: float
    loan_grade_D: float
    loan_grade_E: float
    loan_grade_F: float
    loan_grade_G: float
    cb_person_default_on_file_N: float
    cb_person_default_on_file_Y: float

    def to_dataframe(self):
        return pd.DataFrame([self.model_dump()])


def get_best_threshold():
    with open("../models/best_threshold.json", "r") as file:
        best_threshold = json.load(file)

    return best_threshold["threshold"]


threshold = get_best_threshold()
model = deserialize_data("../models/random_forest_base.pkl").best_estimator_


app = FastAPI()


@app.post("/preprocess")
async def preprocess(item: Item):
    x_data = item.to_dataframe()
    x_data_clean = preprocess_data(x_data).iloc[[0]].to_dict(orient="records")[0]
    return x_data_clean


@app.post("/predict")
async def predict(cleaned_item: CleanedItem):
    cleaned_item = cleaned_item.to_dataframe()
    y_probabilities = model.predict_proba(cleaned_item)[:, 1][0]

    if y_probabilities >= threshold:
        return {"prediction": "yes", "probability": y_probabilities}
    else:
        return {"prediction": "no", "probability": y_probabilities}
