from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model("sentiment_model.h5")
model.save("sentiment_model_new", save_format='tf')
print("Model re-saved successfully!")