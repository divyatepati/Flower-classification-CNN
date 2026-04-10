# predict.py
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/flower_cnn_model.keras"
model = load_model(MODEL_PATH, compile=False)

TRAIN_DIR = "flowers/train"

CLASS_NAMES = [str(i) for i in range(1, 103)]

with open("cat_to_name.json", "r") as f:
    LABEL_MAP = json.load(f)

def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    class_index = int(np.argmax(prediction))
    class_number = CLASS_NAMES[class_index]

    class_name = LABEL_MAP.get(class_number, "Unknown")

    return class_name