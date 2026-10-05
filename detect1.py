import cv2
import time
import os
from ultralytics import YOLO

# ================== MODELS ==================
helmet_model = YOLO("runs/detect/train3/weights/best.pt")
person_model = YOLO("yolo11s.pt")

# ================== VIDEO ==================
cap = cv2.VideoCapture("sample/traffic1.mp4")

# ================== SAVE SETUP ==================
SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_INTERVAL = 5
last_save_time = 0

# ================== SPEED SETTINGS ==================
FRAME_SKIP = 1        # 🔥 biggest speed boost
HELMET_SKIP = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

frame_id = 0

# ================== TRIPLE RIDING ==================
TRIPLE_FRAME_THRESHOLD = 5
triple_counter = {}

def save_full_frame(frame, violation_type):
    global last_save_time
    now = time.time()
    if now - last_save_time >= SAVE_INTERVAL:
        filename = f"{violation_type}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(SAVE_DIR, filename), frame)
        print(f"[SAVED] {filename}")
        last_save_time = now

# ================== MAIN LOOP ==================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # 🔥 FRAME SKIP (KEY FOR SPEED)
    if frame_id % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # ------------------------------------------------
    # NO HELMET (SKIPPED FRAMES)
    # ------------------------------------------------
    if frame_id % HELMET_SKIP == 0:
        helmet_results = helmet_model(frame, conf=0.2)
        no_helmet_detected = False

        for r in helmet_results:
            for box in r.boxes:
                label = helmet_model.names[int(box.cls[0])]
                if label == "no_helmet":
                    no_helmet_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, "NO HELMET",
                                (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0,0,255), 2)

        if no_helmet_detected:
            save_full_frame(frame, "NO_HELMET")

    # ------------------------------------------------
    # PERSON + BIKE TRACKING
    # ------------------------------------------------
    persons = []
    bikes = []

    results = person_model.track(
        frame,
        conf=0.25,
        persist=True,
        tracker="bytetrack.yaml"
    )

    for r in results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()

        for box, cls, tid in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            label = person_model.names[int(cls)]
            tid = int(tid)

            if label == "person":
                persons.append((x1,y1,x2,y2))
            elif label in ["motorbike", "motorcycle"]:
                bikes.append((x1,y1,x2,y2,tid))

    # ------------------------------------------------
    # TRIPLE RIDING (STRICT + TEMPORAL)
    # ------------------------------------------------
    for bx1, by1, bx2, by2, tid in bikes:

        bike_w = bx2 - bx1
        bike_h = by2 - by1
        bike_cx = (bx1 + bx2) // 2

        rider_count = 0

        for px1, py1, px2, py2 in persons:
            pcx = (px1 + px2) // 2

            horizontal_ok = abs(pcx - bike_cx) < bike_w * 0.4

            # 🔥 CRITICAL FIX (prevents false detection)
            foot_overlap = py2 > by1 + bike_h * 0.5 and py2 < by2
            seated_ok = py1 < by1

            if horizontal_ok and foot_overlap and seated_ok:
                rider_count += 1

        if rider_count >= 3:
            triple_counter[tid] = triple_counter.get(tid, 0) + 1
        else:
            triple_counter[tid] = 0

        if triple_counter[tid] >= TRIPLE_FRAME_THRESHOLD:
            cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,0,255), 3)
            cv2.putText(frame, "TRIPLE RIDING / VIOLATION",
                        (bx1, by1-12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,0,255), 3)
            save_full_frame(frame, "TRIPLE_RIDING")

    # ------------------------------------------------
    # DISPLAY
    # ------------------------------------------------
    cv2.imshow("TrafficX – Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()