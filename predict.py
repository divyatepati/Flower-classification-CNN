# predict.py
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = "model/flower_cnn_model.h5"
model = load_model(MODEL_PATH)

# -------------------------
# Get class names EXACTLY like training order
# -------------------------
TRAIN_DIR = "flowers/train"

CLASS_NAMES = sorted([
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

# -------------------------
# Load flower label mapping
# -------------------------
with open("cat_to_name.json", "r") as f:
    LABEL_MAP = json.load(f)

# -------------------------
# Prediction Function
# -------------------------
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    class_index = np.argmax(prediction)

    class_number = CLASS_NAMES[class_index]

    class_name = LABEL_MAP.get(class_number, "Unknown")

    return img_path, class_name