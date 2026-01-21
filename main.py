# main.py

import time
from modules.motion_detection.motion_detector import run_motion_detection
from modules.object_detection.yolo_detector import run_object_detection
from modules.face_detection.face_detector import detect_and_extract_faces
from modules.face_recognition.face_recognizer import (
    initialize_face_database,
    recognize_faces
)
from modules.utils.cleanup import clear_directory

ROI_DIR = "data/processed/roi"


def run_pipeline():
    print("=============================")
    print(" SMART BORDER SECURITY SYSTEM")
    print("=============================")

    # 🔴 Load face database ONCE for entire program
    initialize_face_database()

    print("\nSystem running... Press CTRL+C to stop.\n")

    try:
        while True:
            print("\n=== PHASE 1: Motion Detection ===")
            motion_detected = run_motion_detection()

            if not motion_detected:
                # small sleep to avoid CPU burn
                time.sleep(0.5)
                continue

            print("\n=== PHASE 2: Object Detection ===")
            result = run_object_detection()

            if not result.get("threat_found", False):
                print("No threat detected. Returning to monitoring.")
                time.sleep(0.5)
                continue

            print("\n=== PHASE 3: Face Detection ===")
            face_count = detect_and_extract_faces()
            clear_directory(ROI_DIR)

            if face_count == 0:
                print("No faces extracted. Returning to monitoring.")
                time.sleep(0.5)
                continue

            print("\n=== PHASE 4: Face Recognition ===")
            recognize_faces()

            print("\n✅ Cycle completed. Monitoring resumes...\n")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 System stopped by user. Exiting cleanly.")

if __name__ == "__main__":
    run_pipeline()
