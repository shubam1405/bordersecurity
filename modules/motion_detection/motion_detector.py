import cv2
import time
import os
import sys
from core.logger import logger
from core.exception import BorderSecurityException

def run_motion_detection():
    try:
        logger.info("Starting motion detection module")

        roi_dir = "data/processed/roi"
        os.makedirs(roi_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Camera not accessible")

        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = bg_subtractor.apply(gray)

            # 🔧 Noise removal + region merging
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            valid_boxes = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 3000:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Optional aspect ratio filter
                if h < 80 or w < 80:
                    continue

                valid_boxes.append((x, y, w, h))

            # 🧠 Merge all boxes into ONE (person-level ROI)
            if valid_boxes:
                x_min = min(x for x, y, w, h in valid_boxes)
                y_min = min(y for x, y, w, h in valid_boxes)
                x_max = max(x + w for x, y, w, h in valid_boxes)
                y_max = max(y + h for x, y, w, h in valid_boxes)

                roi = frame[y_min:y_max, x_min:x_max]

                timestamp = int(time.time() * 1000)
                roi_path = os.path.join(roi_dir, f"roi_{timestamp}.jpg")
                cv2.imwrite(roi_path, roi)

                logger.info(f"Merged ROI saved at {roi_path}")

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Motion Detection", frame)
            cv2.imshow("Foreground Mask", fg_mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error("Error in motion detection")
        raise BorderSecurityException(e, sys)

if __name__ == "__main__":
    run_motion_detection()
