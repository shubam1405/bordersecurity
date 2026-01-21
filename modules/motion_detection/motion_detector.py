import cv2
import time
import os
from core.logger import logger

# ================= CONFIG =================
VIDEO_DIR = "data/raw/chunks"
ROI_DIR = "data/processed/roi"

MOTION_THRESHOLD = 5000
RECORD_SECONDS = 5

SHOW_VIDEO = True   # 🔁 set False for production

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)


def run_motion_detection():
    """
    Phase 1:
    - Continuous webcam monitoring
    - Motion → record 5s clip
    - Extract BIG ROI
    - Return True when motion event is processed
    """

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Webcam not accessible")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=50,
        detectShadows=False
    )

    logger.info("🔍 CONTINUOUS MONITORING STARTED (Ctrl+C to stop)")

    recording = False
    out = None
    start_time = None
    video_path = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = bg_subtractor.apply(gray)
            motion_area = cv2.countNonZero(fg_mask)

            # ================= DISPLAY (NO OVERLAP) =================
            if SHOW_VIDEO:
                mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

                # Resize mask to match frame (safety)
                mask_bgr = cv2.resize(mask_bgr, (frame.shape[1], frame.shape[0]))

                combined = cv2.hconcat([frame, mask_bgr])
                cv2.imshow("Live Feed  |  Motion Mask", combined)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ================= MOTION TRIGGER =================
            if motion_area > MOTION_THRESHOLD and not recording:
                timestamp = int(time.time())
                video_path = os.path.join(
                    VIDEO_DIR, f"motion_{timestamp}.mp4"
                )

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    video_path, fourcc, fps, (width, height)
                )

                recording = True
                start_time = time.time()
                logger.info("Motion detected → Recording 5s")

            # ================= RECORD VIDEO =================
            if recording:
                out.write(frame)

                if time.time() - start_time >= RECORD_SECONDS:
                    out.release()
                    recording = False

                    cap.release()
                    if SHOW_VIDEO:
                        cv2.destroyAllWindows()

                    logger.info("Extracting BIG ROI from recorded video")
                    extract_big_roi(video_path)
                    logger.info("BIG ROI extraction completed")

                    return True  # move to Phase 2

    finally:
        cap.release()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()

    return False


def extract_big_roi(video_path):
    """
    Extracts BIG ROIs by merging nearby motion contours
    """

    cap = cv2.VideoCapture(video_path)

    bg = cv2.createBackgroundSubtractorMOG2(
        history=200,
        varThreshold=50,
        detectShadows=False
    )

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg.apply(gray)

        # 🔧 Merge contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        xs, ys, xe, ye = [], [], [], []

        for cnt in contours:
            if cv2.contourArea(cnt) < 3000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        if not xs:
            continue

        x1 = max(min(xs) - 10, 0)
        y1 = max(min(ys) - 10, 0)
        x2 = min(max(xe) + 10, frame.shape[1])
        y2 = min(max(ye) + 10, frame.shape[0])

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        cv2.imwrite(
            os.path.join(ROI_DIR, f"roi_{frame_id}.jpg"),
            roi
        )

    cap.release()