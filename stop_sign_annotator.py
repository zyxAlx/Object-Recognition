import cv2
import numpy as np

class StopSignAnnotator:
    def __init__(self):
        pass

    def annotate(self, frame, stop_signs, frame_area, detector):
        for (x, y, w, h) in stop_signs:
            # Calculate confidence
            confidence = detector.calculate_confidence(w, h, frame_area)

            # Draw rectangle around stop sign
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Write "Stop Sign" and confidence below the rectangle
            label = f"Stop Sign: {confidence:.2f}%"

            # Determine text size and position
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            text_x = x + int((w - text_width) / 2)
            text_y = y + h + 10



            # Draw text
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame
