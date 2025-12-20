import os
import cv2
import numpy as np
from deepface import DeepFace
import logging

# Try sklearn, else fallback to manual cosine
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceRecognition")

FACES_DIR = "data/processed/faces"
FACE_DATABASE_DIR = "data/face_database"
THRESHOLD = 0.45  # similarity threshold


# -----------------------------
# Utility: cosine similarity fallback
# -----------------------------
def cosine_sim(a, b):
    if SKLEARN_AVAILABLE:
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
    else:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# -----------------------------
# Get face embedding safely
# -----------------------------
def get_face_embedding(face_img):
    try:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        result = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            detector_backend="mtcnn",
            enforce_detection=True#If no face is detected → throw error
            
                            #             Why this is important:
                            # Prevents garbage embeddings

                            # Avoids comparing:

                            # Background

                            # Blur

                            # Non‑face objects
        )

        embedding = np.array(result[0]["embedding"])
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    except Exception as e:
        raise RuntimeError("Face not detected")


# -----------------------------
# Build database (SKIPS BAD IMAGES)
# -----------------------------
import json

def build_face_database():
    database = {}

    for person_dir in os.listdir(FACE_DATABASE_DIR):
        full_path = os.path.join(FACE_DATABASE_DIR, person_dir)
        if not os.path.isdir(full_path):
            continue

        info_path = os.path.join(full_path, "info.json")
        if not os.path.exists(info_path):
            logger.warning(f"No info.json for {person_dir}, skipping")
            continue

        with open(info_path, "r") as f:
            person_info = json.load(f)

        embeddings = []

        for img_name in os.listdir(full_path):
            if not img_name.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(full_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            try:
                emb = get_face_embedding(image)
                embeddings.append(emb)
            except:
                logger.warning(f"No face in {img_name}")

        if embeddings:
            database[person_dir] = {
                "info": person_info,
                "embeddings": embeddings
            }
            logger.info(f"Loaded {len(embeddings)} embeddings for {person_info['name']}")

    return database


# -----------------------------
# Recognize faces (SKIPS BAD FACES)
# -----------------------------
def recognize_faces():
    database = build_face_database()

    for face_img in os.listdir(FACES_DIR):
        face_path = os.path.join(FACES_DIR, face_img)
        face = cv2.imread(face_path)

        if face is None:
            logger.warning(f"Could not read face image: {face_img}")
            continue

        try:
            query_emb = get_face_embedding(face)
        except:
            logger.warning(f"No face detected in {face_img}, skipping")
            continue

        best_match = None
        best_score = 0
        for person, embeddings in database.items():
            for person_id, data in database.items():
                for db_emb in data["embeddings"]:
                    score = cosine_sim(query_emb, db_emb)

                    if score > best_score:
                        best_score = score
                        best_match = data["info"]

        if best_score >= THRESHOLD:
            logger.info(
                f"{face_img} → {best_match['name']} "
                f"(DOB={best_match['dob']}, Region={best_match['region']}) ✅ "
                f"(similarity={best_score:.2f})"
            )
        else:
            logger.warning(f"{face_img} → UNKNOWN 🚨")



# -----------------------------
if __name__ == "__main__":
    recognize_faces()
