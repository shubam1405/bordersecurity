from modules.motion_detection.motion_detector import run_motion_detection
from modules.object_detection.yolo_detector import run_object_detection
from modules.face_detection.face_detector import detect_and_extract_faces
from modules.face_recognition.face_recognizer import recognize_faces

print("=============================")
print(" SMART BORDER SECURITY SYSTEM")
print("=============================")

while True:
    motion = run_motion_detection()

    if motion:
        human_found = run_object_detection()

        if human_found:
            face_count = detect_and_extract_faces()

            if face_count > 0:
                recognize_faces()
