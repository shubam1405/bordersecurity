import json
import os

ROI_DIR = "data/processed/roi"
ROI_FILE = os.path.join(ROI_DIR, "roi.json")

class ROIManager:
    def __init__(self):
        os.makedirs(ROI_DIR, exist_ok=True)
        self.roi = self.load()

    def save(self, x1, y1, x2, y2):
        self.roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        with open(ROI_FILE, "w") as f:
            json.dump(self.roi, f)

    def load(self):
        if os.path.exists(ROI_FILE):
            with open(ROI_FILE, "r") as f:
                return json.load(f)
        return None

    def inside(self, bx1, by1, bx2, by2):
        if not self.roi:
            return False

        rx1, ry1, rx2, ry2 = (
            self.roi["x1"], self.roi["y1"],
            self.roi["x2"], self.roi["y2"]
        )

        # Intersection check
        return not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)
