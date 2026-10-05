import cv2
import time
import os
from ultralytics import YOLO

# ================== MODELS ==================
helmet_model = YOLO("runs/detect/train3/weights/best.pt")  # helmet / no_helmet
person_model = YOLO("yolo11s.pt")                          # person + bike

# ================== VIDEO ==================
cap = cv2.VideoCapture("sample/bike2.jpg")  # or 0 for webcam

# ================== SAVE SETUP ==================
SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_INTERVAL = 5  # seconds
last_save_time = 0

# ================== TRIPLE RIDING TEMPORAL LOGIC ==================
TRIPLE_FRAME_THRESHOLD = 5   # consecutive frames needed
triple_counter = {}          # bike_id -> count

def save_frame(frame, violation_type):
    global last_save_time
    current_time = time.time()

    if current_time - last_save_time >= SAVE_INTERVAL:
        filename = f"{violation_type}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        print(f"[SAVED] {path}")
        last_save_time = current_time


# ================== MAIN LOOP ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    # ------------------------------------------------
    # NO HELMET DETECTION
    # ------------------------------------------------
    helmet_results = helmet_model(frame, conf=0.2)
    no_helmet_detected = False

    for r in helmet_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = helmet_model.names[cls]
            conf = box.conf[0]

            if label == "no_helmet":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                no_helmet_detected = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"NO HELMET {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    if no_helmet_detected:
        save_frame(frame, "NO_HELMET")

    # ------------------------------------------------
    # PERSON & BIKE DETECTION
    # ------------------------------------------------
    results = person_model(frame, conf=0.25)

    persons = []
    bikes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = person_model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                persons.append((x1, y1, x2, y2))
            elif label in ["motorbike", "motorcycle"]:
                bikes.append((x1, y1, x2, y2))

    # ------------------------------------------------
    # TRIPLE RIDING (TEMPORAL + CLEAN OUTPUT)
    # ------------------------------------------------
    for i, (bx1, by1, bx2, by2) in enumerate(bikes):

        bike_center_x = (bx1 + bx2) // 2
        bike_width = bx2 - bx1
        riders = []

        for px1, py1, px2, py2 in persons:
            person_center_x = (px1 + px2) // 2

            # horizontal alignment logic
            if abs(person_center_x - bike_center_x) < bike_width // 2:
                riders.append((px1, py1, px2, py2))

        # temporal counting
        if len(riders) >= 3:
            triple_counter[i] = triple_counter.get(i, 0) + 1
        else:
            triple_counter[i] = 0

        # final confirmed violation
        if triple_counter[i] >= TRIPLE_FRAME_THRESHOLD:
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(
                frame,
                "TRIPLE RIDING / VIOLATION",
                (bx1, by1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )
            save_frame(frame, "TRIPLE_RIDING")

    # ------------------------------------------------
    # DISPLAY
    # ------------------------------------------------
    cv2.imshow("TrafficX – Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()