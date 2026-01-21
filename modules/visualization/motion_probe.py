# motion probe


import cv2

class MotionProbe:
    def __init__(self, threshold=4000):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )
        self.threshold = threshold

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = self.bg.apply(gray)
        motion_pixels = cv2.countNonZero(fg)
        return motion_pixels > self.threshold