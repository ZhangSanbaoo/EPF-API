import uvicorn
from fastapi.responses import RedirectResponse
from src.app import get_application
import requests
from fastapi import APIRouter
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import json
import joblib
from sklearn.model_selection import train_test_split
from firestore import FirestoreClient

app = get_application()

@app.get("/")
async def root():
    return RedirectResponse(url = '/docs')

@app.get("/hello")
async def say_hello():
    return {"message": "Hello World"}

router = APIRouter()
@router.get("/download-iris")
async def download_iris():
    url = "https://www.kaggle.com/datasets/uciml/iris/download"
    response = requests.get(url)
    with open('src/data/iris.csv', 'wb') as f:
        f.write(response.content)
    return {"message": "Iris dataset downloaded successfully"}

app.include_router(router, prefix="/data")

@router.get("/load-iris")
async def load_iris():
    df = pd.read_csv('src/data/iris.csv', index=False)
    return df.to_json(orient='records')

app.include_router(router)

@router.post("/process-iris")
async def process_iris():
    df = pd.read_csv('src/data/iris.csv')
    df = df.drop(['Id'], axis=1)
    processed_df = df.dropna()
    processed_df.to_csv('src/data/processed_iris.csv', index=False)
    return processed_df.to_json(orient='records')

app.include_router(router)



@router.post("/split-iris")
async def split_iris():
    df = pd.read_csv('src/data/processed_iris.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Species', axis=1), df['Species'], test_size=0.2)
    train_file = 'src/data/train_iris.csv'
    test_file = 'src/data/test_iris.csv'
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    return {
        "X_train": X_train.to_json(orient='records'),
        "X_test": X_test.to_json(orient='records'),
        "y_train": y_train.to_json(orient='records'),
        "y_test": y_test.to_json(orient='records')
    }

app.include_router(router)

@router.post("/train-model")
async def train_model():
    train_df = pd.read_csv('src/data/train_iris.csv')
    X_train = train_df.drop('Species', axis=1)
    y_train = train_df['Species']
    with open('src/config/model_parameters.json') as f:
        params = json.load(f)
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    joblib.dump(model, 'src/models/iris_model.pkl')
    return {"message": "Model trained and saved successfully"}

app.include_router(router)


from pydantic import BaseModel
import numpy as np

class IrisModel(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@router.post("/predict")
async def predict(iris: IrisModel):
    model = joblib.load('src/models/iris_model.pkl')
    sample_data = np.array([[iris.SepalLengthCm, iris.SepalWidthCm, iris.PetalLengthCm, iris.PetalWidthCm]])
    prediction = model.predict(sample_data)
    return {"prediction": prediction.tolist()}

@router.get("/get-parameters")
def get_parameters():
    firestore_client = FirestoreClient()
    parameters = firestore_client.get("parameters", "parameters")
    return parameters

app.include_router(router)

@app.put("/update-parameters")
async def update_parameters(update_data: dict):
    firestore_client = FirestoreClient()
    firestore_client.update("parameters", "parameters", update_data)
    return {"message": "Parameters updated successfully"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=4200, debug=True, reload=True)
