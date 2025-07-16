from flask import Flask, render_template, request
import os
import numpy as np
import joblib
import tensorflow as tf
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

crop_model = joblib.load("models/crop_model.pkl")
disease_model = tf.keras.models.load_model("models/plant_disease_model.h5")

class_labels = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/crop', methods=['GET', 'POST'])
def crop_form():
    if request.method == 'POST':
        try:
            data = [
                float(request.form.get('nitrogen', 0)),
                float(request.form.get('phosphorus', 0)),
                float(request.form.get('potassium', 0)),
                float(request.form.get('temperature', 0)),
                float(request.form.get('humidity', 0)),
                float(request.form.get('ph', 0)),
                float(request.form.get('rainfall', 0))
            ]
            prediction = crop_model.predict([np.array(data)])[0]
            return render_template("crop_result.html", crop_result=prediction)
        except Exception as e:
            return render_template("crop_result.html", crop_result=f"❌ Error: {e}")
    return render_template("crop_recommendation.html")

@app.route('/disease', methods=['GET', 'POST'])
def disease_form():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template("plant_disease.html", prediction="❌ No image selected.")

        try:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = disease_model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            return render_template("plant_disease.html", prediction=predicted_class, image_file=filepath)
        except Exception as e:
            return render_template("plant_disease.html", prediction=f"❌ Error: {e}")

    return render_template("plant_disease.html")

if __name__ == '__main__':
    app.run(debug=True)
