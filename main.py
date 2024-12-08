import numpy as np
import os
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.preprocessing import image
from prediction import model, label_index, label_mapping, cancerous_classes

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for predictions.
    Uses the uploaded file for making predictions.
    """
    contents = await file.read()

    # Save the file temporarily
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(contents)

    # Load and preprocess the image
    img = image.load_img(temp_file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # The prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = label_index[predicted_class_index]
    predicted_class_name = label_mapping[predicted_class_label]
    probability = float(prediction[0][predicted_class_index])
    probability_percentage = probability * 100
    probability_percentage_str = f"{probability_percentage:.2f}%"

    if predicted_class_label in cancerous_classes:
        cancerous_status = "cancerous"
    else:
        cancerous_status = "non-cancerous"

    os.remove(temp_file_path)  # Clean up the temporary file

    return {
        "prediction": predicted_class_name,
        "probability": probability,
        "probability_percentage": probability_percentage,
        "probability_percentage_str": probability_percentage_str,
        "cancerous_status": cancerous_status,
    }
