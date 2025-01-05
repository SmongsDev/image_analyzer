from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import os

app = Flask(__name__)
CASCADE_FILENAME = 'haarcascade_frontalface_alt.xml'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cascade = cv2.CascadeClassifier(CASCADE_FILENAME)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part in the request."

    file = request.files['file']

    if file.filename == '':
        return "No file selected for uploading."

    if file and allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # 이미지 또는 동영상 처리 분기
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension in {'png', 'jpg', 'jpeg', 'bmp'}:
            processed_path = process_image(filepath)
            return f"<h1>Image Processed Successfully</h1><br><img src='/{processed_path}' />"
        elif file_extension in {'mp4', 'avi'}:
            return redirect(url_for('video_feed', filepath=filepath))

    else:
        return "File type not allowed."

def process_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return "Error: Image not loaded!"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    processed_path = os.path.join(UPLOAD_FOLDER, 'processed_' + os.path.basename(filepath))
    cv2.imwrite(processed_path, image)
    return processed_path

def generate_video(filepath):
    cam = cv2.VideoCapture(filepath)
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

@app.route('/video_feed/<path:filepath>')
def video_feed(filepath):
    full_path = os.path.join(UPLOAD_FOLDER, filepath)
    return Response(generate_video(full_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
