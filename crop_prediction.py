
import joblib
import numpy as np
import os

MODEL_PATH = "models/crop_model.pkl"

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    print("✅ Crop model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load crop model: {str(e)}")
    model = None

def predict_crop(data):
    """
    Predicts the suitable crop for given environmental conditions.
    
    Args:
        data (list): [N, P, K, temperature, humidity, pH, rainfall]
    
    Returns:
        str: Predicted crop name or error message
    """
    if model is None:
        return "❌ Model not available"
    
    try:
        arr = np.array(data).reshape(1, -1)
        prediction = model.predict(arr)
        return prediction[0]
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"
