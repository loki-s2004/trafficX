import cv2
import time

def save_violation(frame, violation_type):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{violation_type}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
