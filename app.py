from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part in the request."

    file = request.files['image']

    if file.filename == '':
        return "No file selected for uploading."

    if file and allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return f"File uploaded successfully: {filepath}"

    else:
        return "File type not allowed."

def process_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return "Error: Image not loaded!"
    
    height, width, _ = image.shape
    print(f"Image Resolution: {width}x{height}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, (800, 600))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    process_path = os.path.join(UPLOAD_FOLDER, 'processed_' + os.path.basename(filepath))
    cv2.imwrite(process_path, image)
    return process_path

if __name__ == '__main__':
    app.run(debug=True)