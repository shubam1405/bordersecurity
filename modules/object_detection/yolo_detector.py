import os
import cv2
from ultralytics import YOLO
from core.logger import logger

# ---------------- PATHS ----------------
ROI_DIR = "data/processed/roi"          # Phase 1 output
OUTPUT_DIR = "data/processed/detections"
MODEL_PATH = "models/yolov8/yolov8n.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO(MODEL_PATH)

# COCO classes we care about
VALID_CLASSES = [
    "person",
    "car", "bus", "truck", "motorcycle",
    "dog", "cat", "cow", "horse", "sheep", "bird",
    "laptop", "tv", "cell phone"
]

# ---------------- MAIN FUNCTION ----------------
def run_object_detection():
    print("Running Phase 2 – Object Detection")

    for img_name in os.listdir(ROI_DIR):
        img_path = os.path.join(ROI_DIR, img_name)

        # Skip directories
        if not os.path.isfile(img_path):
            continue

        # Skip non-image files
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Read image safely
        image = cv2.imread(img_path)

        # Skip corrupted / empty images
        if image is None:
            logger.warning(f"Skipping unreadable file: {img_name}")
            continue

        # Run YOLO inference
        results = model(image, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                confidence = float(box.conf[0])

                if label in VALID_CLASSES:
                    logger.info(
                        f"Detected {label} ({confidence:.2f}) in {img_name}"
                    )

                    save_path = os.path.join(
                        OUTPUT_DIR, f"{label}_{img_name}"
                    )
                    cv2.imwrite(save_path, image)


# ---------------- RUN ----------------
if __name__ == "__main__":
    run_object_detection()
