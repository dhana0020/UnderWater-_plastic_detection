from flask import Flask, render_template, request
import os
from utils import detect_objects
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Call detection logic
    boxes, class_ids, confidences, class_names = detect_objects(filepath)

    # Convert class IDs to class names
    detected = []
    for i in range(len(boxes)):
        detected.append({
            "label": class_names[class_ids[i]],
            "confidence": round(confidences[i] * 100, 2),
            "box": boxes[i]
        })

    return render_template('result.html', image_path=filepath, detections=detected)

if __name__ == '__main__':
    app.run(debug=True)
