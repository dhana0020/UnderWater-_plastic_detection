from ultralytics import YOLO

# Load the YOLO model once when this module is imported
model = YOLO("models/best.pt")


def detect_objects(image_path):
    # Run inference
    results = model(image_path)[0]

    boxes = []
    class_ids = []
    confidences = []

    for box in results.boxes:
        xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0]  # [x1, y1, x2, y2]
        cls = int(box.cls.item())                              # Class ID
        conf = float(box.conf.item())                          # Confidence score

        boxes.append(xyxy)
        class_ids.append(cls)
        confidences.append(conf)

    return boxes, class_ids, confidences, results.names
