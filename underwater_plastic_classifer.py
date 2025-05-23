# -*- coding: utf-8 -*-
"""Copy of plastic_waste_cv.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16Kld8raaiFPkl9lA_zqJfLjlvY7uhS3b
"""

from google.colab import files

# Upload kaggle.json
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d arnavs19/underwater-plastic-pollution-detection --unzip -p /content/

!pip install -q opencv-python-headless joblib

import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

train_path = '/content/underwater_plastics/train'
print("Class folders found in train path:", os.listdir(train_path))

!pip install scikit-image opencv-python-headless -q

import os
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from matplotlib import pyplot as plt
from glob import glob

import os

# Path to the labels folder
label_path = '/content/underwater_plastics/train/labels'

# Initialize an empty set to store class IDs
class_ids = set()

# Loop through all label files in the labels folder
for label_file in os.listdir(label_path):
    label_file_path = os.path.join(label_path, label_file)

    # Read the file
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

        # For each line, get the class ID (first element)
        for line in lines:
            class_id = int(line.strip().split()[0])
            class_ids.add(class_id)

# Print all unique class IDs found
print("Unique class IDs found in dataset:", class_ids)

"""### Building and saving the model"""

# 📂 Define paths
image_path = '/content/underwater_plastics/train/images'
label_path = '/content/underwater_plastics/train/labels'

# 🧠 Class map (YOLO class ids to readable labels)
class_map = {
    0: 'Mask',
    1: 'can',
    2: 'cellphone',
    3: 'electronics',
    4: 'gbottle',
    5: 'glove',
    6: 'metal',
    7: 'misc',
    8: 'net',
    9: 'pbag',
    10: 'pbottle',
    11: 'plastic',
    12: 'rod',
    13: 'sunglasses',
    14: 'tire'
}


# ⚙️ Step 2: Extract features from ROI
def extract_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize ROI to fixed size
    gray = cv2.resize(gray, (64, 64))

    # Compute HOG features
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    # Compute LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    # Color histogram (optional)
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256]*3).flatten()

    # Combine features
    features = np.concatenate([hog_feat, lbp_hist, color_hist])
    return features

# 📊 Step 3: Prepare Dataset
X = []
y = []

for label_file in sorted(glob(os.path.join(label_path, '*.txt'))):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    image_file = os.path.join(image_path, os.path.basename(label_file).replace('.txt', '.jpg'))
    if not os.path.exists(image_file):
        continue

    image = cv2.imread(image_file)
    h, w = image.shape[:2]

    for line in lines:
        class_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
        class_id = int(class_id)

        # Convert normalized bbox to pixel coordinates
        xmin = int((x_center - bbox_w/2) * w)
        ymin = int((y_center - bbox_h/2) * h)
        xmax = int((x_center + bbox_w/2) * w)
        ymax = int((y_center + bbox_h/2) * h)

        # Crop and extract features
        roi = image[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
        if roi.size == 0:
            continue

        features = extract_features(roi)
        X.append(features)
        y.append(class_id)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# 👨‍🏫 Step 4: Train SVM model
print(f"Training samples: {len(y)}, Classes: {set(y)}")
model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
model.fit(X, y)

# 💾 Save model
joblib.dump(model, '/content/plastic_detector_model.pkl')
print("✅ Model trained and saved successfully!")

import os
import cv2
import numpy as np
from glob import glob
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib


image_path = '/content/underwater_plastics/train/images'
label_path = '/content/underwater_plastics/train/labels'


class_map = {
    0: 'Mask', 1: 'can', 2: 'cellphone', 3: 'electronics', 4: 'gbottle',
    5: 'glove', 6: 'metal', 7: 'misc', 8: 'net', 9: 'pbag',
    10: 'pbottle', 11: 'plastic', 12: 'rod', 13: 'sunglasses', 14: 'tire'
}


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    color_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256]*3).flatten()
    return np.concatenate([hog_feat, lbp_hist, color_hist])

X, y = [], []

for label_file in sorted(glob(os.path.join(label_path, '*.txt'))):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    image_file = os.path.join(image_path, os.path.basename(label_file).replace('.txt', '.jpg'))
    if not os.path.exists(image_file):
        continue

    image = cv2.imread(image_file)
    h, w = image.shape[:2]

    for line in lines:
        class_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
        class_id = int(class_id)
        xmin = int((x_center - bbox_w/2) * w)
        ymin = int((y_center - bbox_h/2) * h)
        xmax = int((x_center + bbox_w/2) * w)
        ymax = int((y_center + bbox_h/2) * h)
        roi = image[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]

        if roi.size == 0:
            continue

        features = extract_features(roi)
        X.append(features)
        y.append(class_id)

X = np.array(X)
y = np.array(y)

# 🧪 Split data for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔁 Try different classifiers
classifiers = {
    "SVM": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000),

}

results = {}

for name, clf in classifiers.items():
    print(f"\n🔍 Training {name}...")
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=[class_map[i] for i in sorted(set(y))]))
    results[name] = (model, acc)

# 💾 Save the best model
best_model_name = max(results, key=lambda k: results[k][1])
best_model = results[best_model_name][0]
joblib.dump(best_model, f'/content/best_model_{best_model_name}.pkl')
print(f"\n🎉 Best model: {best_model_name} saved as best_model_{best_model_name}.pkl")

"""### Evaluation using test dataset

### Cross validation
"""

import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt

# Load the trained model
model = joblib.load('/content/best_model_Random Forest.pkl')

# Define class map
class_map = {
    0: 'Mask',
    1: 'can',
    2: 'cellphone',
    3: 'electronics',
    4: 'gbottle',
    5: 'glove',
    6: 'metal',
    7: 'misc',
    8: 'net',
    9: 'pbag',
    10: 'pbottle',
    11: 'plastic',
    12: 'rod',
    13: 'sunglasses',
    14: 'tire'
}

# Function to make predictions on new images
def predict_image(image_path, model):
    # Read the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Dummy detection box (For testing, you can replace this with your actual detection logic)
    # Example: [[xmin, ymin, xmax, ymax]]
    bboxes = [[50, 50, 200, 200]]  # Example bounding box

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox

        # Crop ROI from the image
        roi = image[ymin:ymax, xmin:xmax]
        if roi.size == 0:
            continue

        # Extract features from the ROI
        features = extract_features(roi)

        # Predict the class of the ROI
        prediction = model.predict([features])[0]

        # Draw bounding box and label
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        label = f"{class_map[prediction]}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
image_path = '/content/underwater_plastics/train/images/photo_27_jpg.rf.d29bc019f652e48d8a4bc5b317c49ff6.jpg'  # Replace with your image
predict_image(image_path, model)

image_path = '/test2.jpg'  # Replace with your image
predict_image(image_path, model)

image_path = '/test1.jpg'
predict_image(image_path, model)

