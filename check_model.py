import tensorflow as tf

try:
    model = tf.keras.models.load_model("models/plant_disease_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
