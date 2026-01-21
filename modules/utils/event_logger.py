import json
import os
from datetime import datetime

class EventLogger:
    def __init__(self, log_dir="logs", max_events=50):
        self.max_events = max_events
        self.events = []
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "events.jsonl")

    def log(self, event_type, label, confidence=None, snapshot_path=None):
        event = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "label": label,
            "confidence": confidence,
            "snapshot": snapshot_path
        }

        # UI history
        self.events.insert(0, event)
        if len(self.events) > self.max_events:
            self.events.pop()

        # Persistent log (append-only)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_events(self):
        return self.events