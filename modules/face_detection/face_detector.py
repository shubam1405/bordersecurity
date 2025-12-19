import cv2
import os
import logging

# Paths
DETECTIONS_DIR = "data/processed/detections"
FACES_DIR = "data/processed/faces"

PROTO_PATH = "models/face_detection/deploy.prototxt"
MODEL_PATH = "models/face_detection/res10_300x300_ssd.caffemodel"

CONFIDENCE_THRESHOLD = 0.5

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceDetection")

def load_face_model():
    net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)#You are loading compressed facial knowledge into memory.


        #     What edges look like

        # What eyes look like

        # What noses look like

        # What mouths look like

        # How these features combine into a face
        #This knowledge is encoded as numbers (weights)
    logger.info("Face detection DNN loaded")
    return net

def detect_and_extract_faces():
    os.makedirs(FACES_DIR, exist_ok=True)
    net = load_face_model()
    face_count = 0

    for img_name in os.listdir(DETECTIONS_DIR):
        img_path = os.path.join(DETECTIONS_DIR, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")

                face = image[y1:y2, x1:x2]

                if face.size == 0:
                    continue

                face_count += 1
                face_filename = f"face_{face_count}.jpg"
                save_path = os.path.join(FACES_DIR, face_filename)
                cv2.imwrite(save_path, face)

                logger.info(f"Face extracted → {face_filename}")

    logger.info(f"Total faces extracted: {face_count}")

if __name__ == "__main__":
    print("Running Phase 3 – Face Detection & Extraction")
    detect_and_extract_faces()
