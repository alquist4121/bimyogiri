from typing import Union
from fastapi import FastAPI
from mlmodel import MLModel
import schemas

app = FastAPI()
model = MLModel()
model.load()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/prediction", response_model=schemas.Pred)
async def predict(data: schemas.Data):
    preds = model.predict(data.data)
    return {"prediction": preds}