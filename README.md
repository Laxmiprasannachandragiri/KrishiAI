
# 🌾 KrishiAI – AI-Powered Crop Recommendation & Plant Disease Detection

> A full-stack intelligent agriculture assistant that leverages **Machine Learning** and **Deep Learning** to recommend ideal crops and diagnose plant diseases through image-based detection.

---

## 📌 Project Overview

KrishiAI is an innovative web-based application built for smart farming. It provides:

- 🌱 **Crop Recommendation** based on soil nutrients, weather, and environmental conditions.
- 🦠 **Plant Disease Detection** using Convolutional Neural Networks (CNN) on leaf images.
- 🖥️ A responsive **Flask-powered interface** with a clean AI-themed UI design.

---

## 🚀 Tech Stack

| Frontend | Backend | ML/DL Models | Deployment |
|----------|---------|--------------|------------|
| HTML, CSS, JS | Flask (Python) | RandomForest, CNN (TensorFlow) | Locally / Render / Railway |

---

## 📂 Project Structure

KrishiAI/
├── app.py
├── models/
│ ├── crop_model.pkl # Crop Recommendation model
│ ├── plant_disease_model.h5 # ⚠️ Not pushed due to size limit
├── templates/
│ ├── index.html
│ ├── crop_recommendation.html
│ ├── plant_disease.html
│ ├── crop_result.html
├── static/
│ ├── css/
│ │ └── style.css
│ ├── uploads/
├── crop_recommendation.csv
├── train_crop_model.py
├── train_plant_disease_model.py
├── README.md
├── .gitignore




---

## 🤖 Features

### 1. 🌾 Crop Recommendation
- Input: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall
- Output: Suggested crop best suited for your farm's conditions

### 2. 🦠 Plant Disease Detection
- Input: Leaf image (Tomato plant dataset)
- Output: Predicted disease class with image preview

---

## ⚠️ Notes

- **The `plant_disease_model.h5` file exceeds GitHub’s 100MB file limit**.  
  If needed, it can be downloaded from Google Drive or retrained using `train_plant_disease_model.py`.
- Add the `.h5` file back into the `/models` folder for complete functionality.

---

## 📥 Installation (Local)

```bash
git clone https://github.com/Laxmiprasannachandragiri/KrishiAI.git
cd KrishiAI

# Create a virtual environment
python -m venv venv
venv\Scripts\activate  # For Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py



---

## 🧪 Dataset Sources

- 📊 **Crop Data**: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- 🌿 **Leaf Images**: [PlantVillage Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🎓 Project Submitted To

**Microsoft – AICTE Virtual Internship (2025)**  
By: **Laxmi Prasanna Chandragiri**

---

## 📧 Contact

For any queries or feedback:

- GitHub: [@Laxmiprasannachandragiri](https://github.com/Laxmiprasannachandragiri)
- Email: prasannachandragiri06@gmail.com

