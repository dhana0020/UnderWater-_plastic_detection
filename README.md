# UnderWater-_plastic_detection #

   This project focuses on detecting plastic waste in underwater environments using a hybrid approach of **YOLOv5 object detection** and **machine learning classifiers** like SVM, KNN, and Random Forest. The aim is to assist environmental research and cleanup initiatives by accurately identifying plastic debris in ocean or riverbed images.

## Features ##

-  **YOLOv5 Model** (`best.pt`) for detecting plastic waste in images.
-  **Traditional Classifiers** (`SVM`, `KNN`, `RF`) for further classification of detected objects.
-  **Ensemble Model** (`underwater_pastic_ensemble.py`) to combine deep learning and ML predictions.
-  Modular scripts for detection (`YOLO`), classification (`ML`), and integration (`Ensemble`).
-  Basic app interface setup in `app.py` (can be extended using Flask or Streamlit)
