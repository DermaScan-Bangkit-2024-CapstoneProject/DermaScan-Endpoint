from fastapi import FastAPI, UploadFile
import random

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    """
    Placeholder endpoint for predictions.
    Returns a random binary class (0 or 1) as the prediction.
    """
    # Generate a random binary prediction
    prediction = random.choice([0, 1])
    return {"prediction": prediction, "file": file}
