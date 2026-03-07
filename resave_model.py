from tensorflow.keras.models import load_model

model = load_model("sentiment_model.h5")
model.save("sentiment_model.keras")
print("Model re-saved successfully!")