"""
Description: Flask app for Malaria disease prediction using VGG19 trsansfer learning technique

@author: Kishorlal
"""

import os
import numpy as np
import cv2

# Importing Keras library
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Importing Flask library
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Create flask object
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__),r"templates"))

# Saved path of our model
MODEL_PATH = os.path.join(os.path.dirname(__file__),r"Malaria-Detection-VGG19.h5")

# Load your trained model
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['file']
        basepath = os.path.dirname(__file__)
        # Save the file to be predicted to uploads folder uploads
        image_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(image_path)
        
        # Preprocessing
        img=cv2.imread(image_path)
        img=cv2.resize(img,(224,224))
        img=img/255
        img=np.array(img)
        img=np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img)
        prediction=np.argmax(prediction, axis=1)
        if prediction==0:
            result="The Person is infected with Malaria"
        else:
            result="The Person is not infected with Malaria"
   
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)