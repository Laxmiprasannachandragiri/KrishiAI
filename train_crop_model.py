
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = "crop_recommendation.csv"
MODEL_PATH = "models/crop_model.pkl"

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    data = pd.read_csv(path)
    
    if 'label' not in data.columns:
        raise ValueError("The dataset must contain a 'label' column.")
    
    print("✅ Dataset loaded successfully.")
    return data

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.4f}\n")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def main():
    try:
        data = load_data(DATA_PATH)
        X = data.drop("label", axis=1)
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

        print("\n🌿 Feature Importances:")
        for feat, score in zip(X.columns, model.feature_importances_):
            print(f"{feat}: {score:.4f}")

        save_model(model, MODEL_PATH)

    except Exception as e:
        print("❌ Error during training:", e)

if __name__ == "__main__":
    main()
