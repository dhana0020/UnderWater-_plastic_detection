from ultralytics import YOLO
import cv2  # Optional: remove if not used elsewhere

# Load the YOLO model once
model = YOLO("models/best.pt")  # Ensure this path is correct

def detect_objects(image_path):
    # Run inference
    results = model(image_path)[0]

    boxes = []
    class_ids = []
    confidences = []

    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]  # [x1, y1, x2, y2]
        cls = int(box.cls.item())
        conf = float(box.conf.item())

        boxes.append(tuple(xyxy))
        class_ids.append(cls)
        confidences.append(conf)

    class_names = results.names  # or model.names — both are fine

    return boxes, class_ids, confidences, class_names
