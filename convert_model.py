from tensorflow.keras.models import load_model

# Load old model
model = load_model("model/flower_cnn_model.h5")

# Save in new format
model.save("model/flower_cnn_model.keras")

print("Conversion done! New .keras model created.")