import os
import cv2
import json
import numpy as np
from deepface import DeepFace
from core.logger import logger

FACE_DATABASE_DIR = "data/face_database"
FACES_DIR = "data/processed/faces"
THRESHOLD = 0.75

DATABASE = None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_face_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = DeepFace.represent(
        img_path=img,
        model_name="Facenet",
        detector_backend="mtcnn",
        enforce_detection=True
    )
    emb = np.array(result[0]["embedding"], dtype="float32")
    return emb / np.linalg.norm(emb)


def build_face_database():
    database = []

    for person in os.listdir(FACE_DATABASE_DIR):
        person_dir = os.path.join(FACE_DATABASE_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        info_path = os.path.join(person_dir, "info.json")
        if not os.path.exists(info_path):
            continue

        with open(info_path) as f:
            info = json.load(f)

        embeddings = []
        for file in os.listdir(person_dir):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img = cv2.imread(os.path.join(person_dir, file))
            if img is None:
                continue

            try:
                embeddings.append(get_face_embedding(img))
            except:
                pass

        if embeddings:
            database.append({"info": info, "embeddings": embeddings})
            logger.info(f"Loaded embeddings for {info['name']}")

    logger.info(f"Known identities loaded: {len(database)}")
    return database


def recognize_faces():
    global DATABASE

    logger.info("Starting Face Recognition")

    if DATABASE is None:
        DATABASE = build_face_database()

    if not os.listdir(FACES_DIR):
        logger.warning("No extracted faces to recognize")
        return

    for face_file in os.listdir(FACES_DIR):
        face = cv2.imread(os.path.join(FACES_DIR, face_file))
        if face is None:
            continue

        try:
            query = get_face_embedding(face)
        except:
            continue

        best_score = 0
        best_person = None

        for person in DATABASE:
            for emb in person["embeddings"]:
                score = cosine_similarity(query, emb)
                if score > best_score:
                    best_score = score
                    best_person = person["info"]

        if best_score >= THRESHOLD:
            logger.critical(
                f"✅ IDENTIFIED: {best_person['name']} | Similarity={best_score:.2f}"
            )
        else:
            logger.warning(
                f"❌ UNKNOWN FACE | Similarity={best_score:.2f}"
            )
