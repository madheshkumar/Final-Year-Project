import base64
import io
import cv2 as cv
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import predict
import enhance



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def run():
    
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']
    
    
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    
    pil_image = Image.open(file)
    
    # Convert the PIL image to base64 format
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    original_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    enchancedImaged = enhance.enhancer(file)
    predictedOutput = predict.prediction(enchancedImaged)

    # Encode the CLAHE output image as base64
    clahe_output = np.array(enchancedImaged)
    bytes = cv.imencode('.png', clahe_output)[1].tobytes()
    clahe_base64 = base64.b64encode(bytes).decode('utf-8')

    
    return render_template('index.html', original_image = original_base64,clahe_image=clahe_base64, prediction = predictedOutput)
 

if __name__ == '__main__':
    app.run(debug=True)
