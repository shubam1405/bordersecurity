# RUN WITH:
# streamlit run modules/visualization/dashboard.py

import sys, os, time, csv
sys.path.append(os.getcwd())

import streamlit as st
import cv2
from datetime import datetime

from modules.object_detection.yolo_detector import YOLODetector
from modules.alarm.alarm_controller import (
    start_alarm, stop_alarm, enable_alarm, is_alarm_enabled
)
from modules.utils.event_logger import EventLogger
from modules.utils.evidence_manager import EvidenceManager

# ================= CONFIG =================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
CSV_LOG_PATH = os.path.join(LOG_DIR, "events.csv")

# ================= STREAMLIT =================
st.set_page_config(page_title="Smart Border Intrusion System", layout="wide")
st.title("🚨 Smart Border Intrusion Detection System")

# ================= SESSION STATE =================
if "source" not in st.session_state:
    st.session_state.source = "Live Camera"

if "yolo" not in st.session_state:
    st.session_state.yolo = YOLODetector("yolov8n.pt", conf=0.5)

if "event_logger" not in st.session_state:
    st.session_state.event_logger = EventLogger(max_events=50)

if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = {}

if "evidence" not in st.session_state:
    st.session_state.evidence = EvidenceManager()

# ================= SIDEBAR =================
st.sidebar.header("⚙ Controls")

source = st.sidebar.radio("Video Source", ["Live Camera", "Demo Video"])

if source != st.session_state.source:
    stop_alarm()
    st.session_state.evidence.stop()
    st.session_state.source = source

enable_alarm(
    st.sidebar.toggle("🔊 Alarm Enabled", value=is_alarm_enabled())
)

# ================= DEMO VIDEO =================
demo_video_path = None
if source == "Demo Video":
    demo_dir = "demo_videos"
    videos = [v for v in os.listdir(demo_dir) if v.endswith((".mp4", ".avi"))]
    demo_video = st.sidebar.selectbox("Select Demo Video", videos)
    demo_video_path = os.path.join(demo_dir, demo_video)

# ================= HISTORY PANEL =================
with st.sidebar.expander("📜 Detection History", expanded=True):
    history_box = st.empty()

# ================= VIDEO =================
cap = cv2.VideoCapture(0 if source == "Live Camera" else demo_video_path)

frame_box = st.empty()
status_box = st.empty()

# ================= CSV INIT =================
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, "w", newline="") as f:
        csv.writer(f).writerow(["time", "type", "label", "confidence"])

def persist_event(event_type, label, confidence=None):
    with open(CSV_LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type,
            label,
            confidence if confidence else ""
        ])

def should_log(label, cooldown=2):
    now = time.time()
    last = st.session_state.last_log_time.get(label, 0)
    if now - last > cooldown:
        st.session_state.last_log_time[label] = now
        return True
    return False

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = st.session_state.yolo.detect(frame)
    intrusion_detected = False

    for label, conf, x1, y1, x2, y2, category in detections:

        if category == "intrusion":
            intrusion_detected = True
            color = (0, 0, 255)
            text = f"INTRUSION: {label.upper()} {conf:.2f}"
            event_type = "INTRUSION"

        elif category == "animal":
            color = (0, 255, 0)
            text = f"ANIMAL: {label.upper()}"
            event_type = "ANIMAL"

        else:
            color = (255, 165, 0)
            text = f"OBJECT: {label.upper()}"
            event_type = "OBJECT"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if should_log(label):
            st.session_state.event_logger.log(event_type, label, conf)
            persist_event(event_type, label, conf)

    # ================= ALARM + EVIDENCE =================
    if intrusion_detected:
        start_alarm()
        status_box.error("🚨 INTRUSION DETECTED")

        if not st.session_state.evidence.recording:
            st.session_state.evidence.start(frame)

        st.session_state.evidence.write(frame)

    else:
        stop_alarm()
        st.session_state.evidence.stop()
        status_box.success("✅ Area Secure")

    # ================= HISTORY =================
    with history_box.container():
        for e in st.session_state.event_logger.get_events():
            st.markdown(
                f"**{e['time']}** | `{e['type']}` | {e['label']} "
                + (f"({e['confidence']:.2f})" if e['confidence'] else "")
            )

    frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    time.sleep(0.03)

cap.release()
st.session_state.evidence.stop()
stop_alarm()