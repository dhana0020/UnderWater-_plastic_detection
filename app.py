import gradio as gr
from utils import detect_objects
from PIL import Image
import numpy as np
import cv2

def detect_and_annotate(image):
    # Save PIL image temporarily
    temp_path = "temp.jpg"
    image.save(temp_path)

    # Run detection
    boxes, class_ids, confidences, class_names = detect_objects(temp_path)

    # Convert PIL to OpenCV image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    # Convert back to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

demo = gr.Interface(
    fn=detect_and_annotate,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="numpy"),
    title="Underwater Plastic Detection",
    description="Upload an image to detect plastic using a YOLO model (best.pt)"
)

if __name__ == "__main__":
    demo.launch()
