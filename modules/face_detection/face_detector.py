import cv2
import os
from core.logger import logger

PERSON_DIR = "data/processed/detection/person"
FACE_DIR = "data/processed/faces"

os.makedirs(FACE_DIR, exist_ok=True)

def detect_and_extract_faces():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    count = 0

    for img_name in os.listdir(PERSON_DIR):
        img_path = os.path.join(PERSON_DIR, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            cv2.imwrite(f"{FACE_DIR}/face_{count}.jpg", face)
            count += 1

    logger.info(f"Total faces extracted: {count}")
    return count
