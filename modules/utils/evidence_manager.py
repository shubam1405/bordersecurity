import os
import cv2
import json
from datetime import datetime


class EvidenceManager:
    def __init__(self, base_dir="evidence"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.recording = False
        self.video_writer = None
        self.event_dir = None

    def start(self, frame, metadata=None):
        """
        Start evidence recording.
        Saves:
        - evidence.avi (video)
        - snapshot.jpg
        - metadata.json
        """
        if self.recording:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_dir = os.path.join(self.base_dir, f"intrusion_{timestamp}")
        os.makedirs(self.event_dir, exist_ok=True)

        # ---------------- VIDEO (FIXED CODEC) ----------------
        h, w, _ = frame.shape
        video_path = os.path.join(self.event_dir, "evidence.avi")

        self.video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"XVID"),  # ✅ Highly compatible
            20,
            (w, h)
        )

        if not self.video_writer.isOpened():
            raise RuntimeError("❌ VideoWriter failed to open")

        # ---------------- SNAPSHOT ----------------
        snapshot_path = os.path.join(self.event_dir, "snapshot.jpg")
        cv2.imwrite(snapshot_path, frame)

        # ---------------- METADATA ----------------
        final_metadata = {
            "event": "INTRUSION",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video": "evidence.avi",
            "snapshot": "snapshot.jpg"
        }

        if metadata:
            final_metadata.update(metadata)

        with open(os.path.join(self.event_dir, "metadata.json"), "w") as f:
            json.dump(final_metadata, f, indent=4)

        self.recording = True

    def write(self, frame):
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

    def stop(self):
        if self.recording and self.video_writer:
            self.video_writer.release()

        self.recording = False
        self.video_writer = None
        self.event_dir = None
