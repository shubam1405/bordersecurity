import cv2
import time
import os
from core.logger import logger

VIDEO_DIR = "data/raw/chunks"
ROI_DIR = "data/processed/roi"

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)


def run_motion_detection():
    cap = cv2.VideoCapture(0)
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40)

    logger.info("🔍 CONTINUOUS MONITORING STARTED (Ctrl+C to stop)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg = bg.apply(frame)
        motion_area = cv2.countNonZero(fg)

        if motion_area > 5000:
            logger.info("Motion detected → Recording 5s")
            record_and_extract_roi(cap)
            return True  # 🔑 tells main.py motion happened

        cv2.waitKey(1)


def record_and_extract_roi(cap):
    timestamp = int(time.time())
    video_path = f"{VIDEO_DIR}/motion_{timestamp}.mp4"

    fps = 20
    h, w, _ = cap.read()[1].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    start = time.time()
    frames = []

    while time.time() - start < 5:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames.append(frame)

    out.release()

    extract_big_roi(frames)


def extract_big_roi(frames):
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30)
    logger.info("Extracting BIG ROI from recorded video")

    idx = 0
    for frame in frames:
        fg = bg.apply(frame)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 8000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        path = f"{ROI_DIR}/roi_{idx}.jpg"
        cv2.imwrite(path, roi)
        idx += 1

    logger.info("BIG ROI extraction completed")
