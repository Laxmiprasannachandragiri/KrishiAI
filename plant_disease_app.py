
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import os

MODEL_PATH = "models/plant_disease_model.h5"

class_labels = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Plant disease model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load plant disease model: {e}")
    model = None

def predict_disease(img_path):
    """
    Predicts the disease of a plant leaf image.

    Args:
        img_path (str): File path to the leaf image.

    Returns:
        str: Predicted disease label or error message.
    """
    if model is None:
        return "❌ Model not available"

    if not os.path.exists(img_path):
        return f"❌ Image not found at path: {img_path}"

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]
        return predicted_class

    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"
