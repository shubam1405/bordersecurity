import cv2
import time
import os
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
            detectShadows=True
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = bg_subtractor.apply(gray)

            # Remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                if cv2.contourArea(cnt) < 1000:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                roi = frame[y:y+h, x:x+w]

                timestamp = int(time.time() * 1000)
                roi_path = os.path.join(roi_dir, f"roi_{timestamp}.jpg")

                cv2.imwrite(roi_path, roi)
                logger.info(f"ROI saved at {roi_path}")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


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
