from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

def process_image(file):
    image = Image.open(file)
    image = image.convert('RGB')  # Ensures consistency
    image = image.resize((224, 224))  # Or whatever your model expects
    return image

allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Make sure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using MobileNetV2
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]  # Top 3 results

        # Format predictions: [(label, percent), ...]
        formatted_preds = [(label, round(prob * 100, 2)) for (_, label, prob) in decoded]

        return render_template('index.html', predictions=formatted_preds)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
