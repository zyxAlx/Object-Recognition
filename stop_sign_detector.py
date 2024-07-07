import cv2

class StopSignDetector:
    def __init__(self, cascade_path):
        self.stop_sign_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stop_signs = self.stop_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=35, minSize=(30, 30))
        return stop_signs

    def calculate_confidence(self, w, h, frame_area):
        sign_area = w * h
        confidence = ((sign_area / frame_area) * 100)
        confidence=min(max(confidence, 0.0), 0.99)
        return confidence