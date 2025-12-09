import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("hand_numbers_model.keras")  # or .h5

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("model_numbers.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as model_numbers.tflite")
