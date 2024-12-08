import os
import hashlib
import gdown
from tensorflow.keras.models import load_model

MODEL_FILE = "model.keras"
MODEL_GDRIVE_ID = "1qEI6UVtlVHUi9EmpAaIztk5SWJqk66KS"
MD5_CHECKSUM = "122c7b5779cfe0acc44a3f59e44229ea"

label_mapping = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BCC": "Basal Cell Carcinoma",
    "AKIEC": "Actinic Keratoses and Bowens Disease",
    "BKL": "Benign Keratosis-like Lesions",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesions",
}

label_index = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

cancerous_classes = {"MEL", "BCC", "AKIEC"}
non_cancerous_classes = {"NV", "BKL", "DF", "VASC"}


def download_model():
    gdown.download(id=MODEL_GDRIVE_ID, output=MODEL_FILE, quiet=False)


def verify_md5(file_path, expected_md5):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5


def load_keras_model():
    if not os.path.exists(MODEL_FILE) or not verify_md5(MODEL_FILE, MD5_CHECKSUM):
        download_model()
        if not verify_md5(MODEL_FILE, MD5_CHECKSUM):
            raise ValueError("Downloaded model file is corrupted.")
    return load_model(MODEL_FILE)


model = load_keras_model()
