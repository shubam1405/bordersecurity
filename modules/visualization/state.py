# modules/visualization/state.py

import time

class AppState:
    def __init__(self):
        self.current_mode = "video"   # video | motion | person
        self.motion_detected = False
        self.person_detected = False
        self.last_alarm_time = 0

        # USER CONTROLS
        self.alarm_enabled = True      # 🔔 TOGGLE
        self.cooldown_seconds = 5

state = AppState()