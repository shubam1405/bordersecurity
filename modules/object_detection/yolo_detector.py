import os
import cv2
import shutil
from ultralytics import YOLO
from core.logger import logger
from modules.alarm.alarm import trigger_alarm

ROI_DIR = "data/processed/roi"
DET_DIR = "data/processed/detection"

CONF_TH = 0.85

ANIMALS = {"cat", "dog", "cow", "horse", "sheep", "bird"}

def ensure_dirs():
    for d in ["person", "cat", "dog", "cow", "other"]:
        os.makedirs(os.path.join(DET_DIR, d), exist_ok=True)

def run_object_detection():
    """
    Phase 2 – Object Detection
    Returns:
        {
            "threat_found": bool,
            "person_found": bool
        }
    """
    ensure_dirs()
    logger.info("Running Phase 2 – Object Detection")

    model = YOLO("yolov8n.pt")

    threat_found = False
    person_found = False

    for name in os.listdir(ROI_DIR):
        path = os.path.join(ROI_DIR, name)

        if not os.path.exists(path):
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        results = model(img, verbose=False)
        detected_any = False

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                if conf < CONF_TH:
                    continue

                detected_any = True

                # 🧍 PERSON → THREAT
                if label == "person":
                    save_path = os.path.join(DET_DIR, "person", name)
                    shutil.copy(path, save_path)

                    logger.critical(f"🚨 PERSON DETECTED ({conf:.2f}) → {name}")
                    trigger_alarm(img)

                    threat_found = True
                    person_found = True

                # 🐕 ANIMALS → NOT A THREAT
                elif label in ANIMALS:
                    save_path = os.path.join(DET_DIR, label, name)
                    shutil.copy(path, save_path)
                    logger.info(f"Animal detected ({label})")

                # 📦 OTHER OBJECTS
                else:
                    save_path = os.path.join(DET_DIR, "other", name)
                    shutil.copy(path, save_path)
                    logger.info(f"Object detected ({label})")

        # Nothing detected at all
        if not detected_any:
            shutil.copy(path, os.path.join(DET_DIR, "other", name))

    return {
        "threat_found": threat_found,
        "person_found": person_found
    }
