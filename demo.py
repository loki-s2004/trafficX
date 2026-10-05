import cv2
import time
import os
from ultralytics import YOLO

# =====================================================
# LOAD MODELS
# =====================================================
helmet_model = YOLO("runs/detect/train3/weights/best.pt")
person_model = YOLO("yolo11s.pt")

# =====================================================
# SAVE FOLDER
# =====================================================
SAVE_DIR = "violations"
os.makedirs(SAVE_DIR, exist_ok=True)

last_save_time = 0
SAVE_INTERVAL = 5

# =====================================================
# VIDEO SETTINGS
# =====================================================
FRAME_SKIP = 1
HELMET_SKIP = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

frame_id = 0
TRIPLE_FRAME_THRESHOLD = 5
triple_counter = {}

# =====================================================
# SAVE FUNCTION
# =====================================================
def save_full_frame(frame, violation_type):
    global last_save_time
    now = time.time()

    if now - last_save_time >= SAVE_INTERVAL:
        filename = f"{violation_type}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        print(f"[SAVED] {path}")
        last_save_time = now

# =====================================================
# IMAGE MODE
# =====================================================
def run_image_mode(path):

    img = cv2.imread(path)

    if img is None:
        print("Image not found")
        return

    # ---------------- PERSON + BIKE ----------------
    results = person_model(img, conf=0.25)

    persons = []
    bikes = []

    for r in results:
        for box in r.boxes:

            label = person_model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "person":
                persons.append((x1, y1, x2, y2))

            elif label in ["motorbike", "motorcycle"]:
                bikes.append((x1, y1, x2, y2))

    # ---------------- TRIPLE RIDING ----------------
    valid_riders = []

    for bx1, by1, bx2, by2 in bikes:

        bike_w = bx2 - bx1
        bike_h = by2 - by1
        bike_cx = (bx1 + bx2) // 2
        bike_cy = (by1 + by2) // 2

        riders = []

        for px1, py1, px2, py2 in persons:

            pcx = (px1 + px2) // 2
            pcy = (py1 + py2) // 2

            horizontal_ok = abs(pcx - bike_cx) < bike_w * 0.6

            vertical_ok = (
                py2 > by1 - bike_h * 0.3 and
                py1 < by2
            )

            center_ok = abs(pcy - bike_cy) < bike_h

            if horizontal_ok and vertical_ok and center_ok:
                riders.append((px1, py1, px2, py2))

        valid_riders.extend(riders)

        if len(riders) >= 3:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(
                img,
                "TRIPLE RIDING / VIOLATION",
                (bx1, by1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

    # ---------------- NO HELMET ----------------
    helmet_results = helmet_model(img, conf=0.2)

    for r in helmet_results:
        for box in r.boxes:

            label = helmet_model.names[int(box.cls[0])]

            if label == "no_helmet":

                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])

                is_rider = False

                for px1, py1, px2, py2 in valid_riders:
                    if hx1 > px1 and hx2 < px2 and hy2 > py1:
                        is_rider = True
                        break

                if not is_rider:
                    continue

                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    "NO HELMET",
                    (hx1, hy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # ---------------- DISPLAY FIXED ----------------
    img = cv2.resize(img, (900, 600))

    cv2.namedWindow("TrafficX Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("TrafficX Detection", img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cv2.destroyAllWindows()

# =====================================================
# VIDEO / WEBCAM MODE
# =====================================================
def run_video_mode(source):

    global frame_id

    cap = cv2.VideoCapture(source)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        if frame_id % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # ---------------- NO HELMET ----------------
        if frame_id % HELMET_SKIP == 0:

            helmet_results = helmet_model(frame, conf=0.2)
            no_helmet_detected = False

            for r in helmet_results:
                for box in r.boxes:

                    label = helmet_model.names[int(box.cls[0])]

                    if label == "no_helmet":

                        no_helmet_detected = True

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

                        cv2.putText(
                            frame,
                            "NO HELMET",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,0,255),
                            2
                        )

            if no_helmet_detected:
                save_full_frame(frame, "NO_HELMET")

        # ---------------- PERSON + BIKE TRACKING ----------------
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

        # ---------------- TRIPLE RIDING ----------------
        for bx1, by1, bx2, by2, tid in bikes:

            bike_w = bx2 - bx1
            bike_h = by2 - by1
            bike_cx = (bx1 + bx2) // 2

            rider_count = 0

            for px1, py1, px2, py2 in persons:

                pcx = (px1 + px2) // 2

                horizontal_ok = abs(pcx - bike_cx) < bike_w * 0.4
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

                cv2.putText(
                    frame,
                    "TRIPLE RIDING / VIOLATION",
                    (bx1, by1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,0,255),
                    3
                )

                save_full_frame(frame, "TRIPLE_RIDING")

        cv2.imshow("TrafficX Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================================
# MENU
# =====================================================
print("1 = IMAGE")
print("2 = VIDEO")
print("3 = WEBCAM")

choice = input("Enter choice: ")

if choice == "1":

    path = input("Enter image path: ")
    run_image_mode(path)

elif choice == "2":

    path = input("Enter video path: ")
    run_video_mode(path)

elif choice == "3":

    run_video_mode(0)

else:
    print("Invalid Choice")